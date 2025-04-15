from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import csv
from matplotlib import image
from PIL import Image
import math
import glob
from sklearn.metrics import confusion_matrix
import getpass
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, jaccard_score
import itertools
import re
from sklearn.metrics import classification_report

import logging
temp_time ='{:04}_{:02}_{:02}__{:02}_{:02}_{:02}'.format(time.gmtime().tm_year, time.gmtime().tm_mon, time.gmtime().tm_mday,  time.gmtime().tm_hour, time.gmtime().tm_min, time.gmtime().tm_sec)
logging.basicConfig(filename='./logs/{}_logfile.log'.format(temp_time), format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.info('start logging')

__author__ = "Raphael Kronberg Department of Molecular Medicine II, Medical Faculty," \
             " Heinrich-Heine-University"
__license__ = "MIT"
__version__ = "1.0.1"
__status__ = "Prototype 2021: This progam/code can not be used as diagnostic tool."
__credits__ = "Pls cite and refer to when using the code: Paweł Stachura, Zhe Lu, Raphael M. Kronberg et al. Deep transfer learning approach for automated cell death classification reveals novel ferroptosis-inducing agents in subsets of B-ALL. , Cell Death and Disease, 2025"

# seed
torch.manual_seed(12345)
np.random.seed(12345)
# max pixel increase
Image.MAX_IMAGE_PIXELS = None

# based on https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

class FromDict2Dataset(torch.utils.data.Dataset):
    '''
    Dataset class for the analytic task dictionary
    '''

    def __init__(self, data_dict, phase, class_names, transform=None):
        '''
        :param data_dict: dict(dict_train, dict_val, dict_test, dict_infer), dict_X[label_i] = data
        :param modus: 'train' or 'val' or 'test' or 'ana'
        :param transform: transformation for the images
        :param labels: list of the labels as strings
        :type data_dict: dictionary of dictionaries
        :type modus: string
        :type transform: transform
        :type labels: list of strings
        :return: get_item return data / label pairs
        :rtype: input for torch.dataloader
        '''
        self.data_dict = data_dict
        self.phase = phase
        self.transform = transform
        self.labels = class_names
        self.__maplabels__()
        self.__createdatalist__()

    def __maplabels__(self):
        ''' map the label stings in class numbers '''
        self.label_map = {}
        counter = 0
        for ent in self.labels:
            self.label_map[ent] = counter
            counter += 1
        print(self.label_map)
        logging.info(self.label_map)

    def __createdatalist__(self):
        ''' Create from a input dict(dict) the corresponding data_list for the modus with pairs of data/label'''
        self.data_list = []
        for i in self.data_dict[self.phase].keys():
            data_values_list = self.data_dict[self.phase][i]
            for entry in data_values_list:
                self.data_list.append([entry, self.label_map[i]])
        print('data samples', self.phase, len(self.data_list))

    def __getitem__(self, index):
        ''' custom get_item function for torch dataloader '''
        data_tuple = self.data_list[index]
        data_image = Image.fromarray(data_tuple[0])
        data_label = data_tuple[1]
        if self.transform:
            data_image = self.transform(data_image)
        return data_image, data_label

    def __len__(self):
        return len(self.data_list)


########################################################################################################################
class Trainer():
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.mod = self.get_os()
        self.train_tile_path = args.train_tile_path
        self.train_val_test = args.train_val_test
        self.train_tile_classes = args.train_tile_classes
        self.class_names = args.class_names
        self.tile_size = args.tile_size
        self.dont_save_tiles = args.dont_save_tiles
        self.model_name = args.model_name
        self.num_classes = len(args.train_tile_classes)
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_train_layers = args.num_train_layers
        self.feature_extract = args.feature_extract
        self.use_pretrained = args.use_pretrained
        self.optimizer_name = args.optimizer_name
        self.criterion_name = args.criterion_name
        self.scheduler_name = args.scheduler_name
        self.device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
        self.img_data_dict = args.img_data_dict
        self.input_size = args.input_size
        self.reload_path = args.reload_path
        self.folder_path = args.folder_path
        self.learning_rate = args.learning_rate
        self.is_inception = False
        self.num_images = 10
        #
        self.model_ft = None
        self.dataloader_dict = None
        self.optimizer_ft = None
        self.criterion_ft = None
        self.scheduler_ft = None
        #
        self.reload = args.reload
        self.hist = None
        self.save_path = args.save_path
        self.params_to_update = None
        self.result_file_name = args.result_file_name
        self.model_id = args.model_id
        self.resize = False
        self.resize_shape = [4032,4032,3]
        self.selection = False #
        self.early_stop = args.early_stop
        self.pixel_cutoff = args.pixel_cutoff
        self.tissue_per = args.tissue_per
        self.gamma = args.gamma
        self.lr_step_size = args.lr_step_size
        self.normalize_on = args.normalize_on
        self.data_load_shuffle = args.data_load_shuffle
        self.label_coloring =args.label_coloring
        self.save_colored_dir = args.save_colored_dir
        self.session_id = args.session_id
        self.patho_cut_off =args.patho_cut_off
        self.reject_counter = 0
        self.accept_counter = 0
        self.check = self.args_printer(args)
        self.one_color_channel = args.one_color_channel
        pass

    def args_printer(self, args):
        for ent in args.__dict__:
            logging.info((ent, args.__dict__[ent]))
        return None

    def model_maker(self):
        '''
        Init the model, the dataloader, the optimizer, the loss function adn the lr sheduler
        :param reload: switch for analyze and training
        :type reload: boolean
        :return: model_ft, dataloader_dict, optimizer_ft, criterion_ft, scheduler_ft
        :rtype:  torch.model, torch.dataloader, torch.optimizer, torch.loss, torch.scheduler
        '''
        self.print_cuda()
        if self.dont_save_tiles == 'yes' and not self.reload:
            self.get_that_tiles()
        else:
            self.img_data_dict = None

        # Initialize the model for this run
        self.initialize_model()

        if not self.reload:
            self.initialize_dataloader()

        # Send the model to GPU
        self.model_ft.to(self.device)

        self.select_retrained_children()

        self.update_params()

        # Observe that all parameters are being optimized
        self.initialize_optimizer()

        # Setup the loss fxn
        self.initialize_criterion()
        # Learning rate scheduler
        self.initialize_scheduler()


    def model_train(self):
        '''
        Init the model for training
        '''
        # create the model
        self.model_maker()
        # train the model
        self.train_model()

    def reload_model(self):
        '''
        Load the model for analyze
        '''
        # init the model
        self.model_maker()
        # load parameters from the pth file
        try:
            self.model_ft.load_state_dict(torch.load(self.reload_path))
        except Exception as e:
            print(e)
            print("Map device to CPU")
            self.model_ft.load_state_dict(torch.load(self.reload_path, map_location=torch.device('cpu')))
        self.model_ft.eval()

    ########################################################################################################################
    def train_model(self):
        '''
        (Re-)training the model
        '''
        #print('rate accepted ', self.accept_counter / (self.accept_counter + self.reject_counter))
        since = time.time()
        val_acc_history = []
        best_model_wts = copy.deepcopy(self.model_ft.state_dict())
        best_acc = 0.0
        # params
        not_improved_count = 0
        test_data_store = []
        # create the log file
        with open('./logs/train_val_test_log_{}.csv'.format(self.datestr()), 'w', newline='') as csvfile:
            logging.info(('./logs/train_val_test_log_{}.csv'.format(self.datestr())))
            fieldnames = ['epoch', 'phase', 'loss', 'accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

            writer.writeheader()

            best_epoch = None
            for epoch in range(self.num_epochs):
                print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
                print('-' * 10)
                logging.info(('Epoch {}/{}'.format(epoch, self.num_epochs - 1)))
                logging.info(('-' * 10))
                # Each epoch has a training and validation phase
                for phase in ['train', 'val', 'test']:
                    print('{}:'.format(phase))
                    if phase == 'train':
                        self.model_ft.train()  # Set model to training mode
                    else:
                        self.model_ft.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    progress_counter = 0
                    for inputs, labels in self.dataloader_dict[phase]:
                        # print(inputs, labels)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        progress_counter +=1
                        print(f"Progress {progress_counter}")
                        # zero the parameter gradients
                        self.optimizer_ft.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Get model outputs and calculate loss
                            # Special case for inception because in training it has an auxiliary output. In train
                            #   mode we calculate the loss by summing the final output and the auxiliary output
                            #   but in testing we only consider the final output.
                            if self.is_inception and phase == 'train':
                                # From https://discuss.pytorch.org/t/how-to-optimize-i
                                # nception-model-with-auxiliary-classifiers/7958
                                outputs, aux_outputs = self.model_ft(inputs)
                                loss1 = self.criterion_ft(outputs, labels)
                                loss2 = self.criterion_ft(aux_outputs, labels)
                                loss = loss1 + 0.4 * loss2
                            else:

                                outputs = self.model_ft(inputs)
                                # outputs = logsoft(outputs) #not if CEL
                                loss = self.criterion_ft(outputs, labels)

                            _, preds = torch.max(outputs, 1)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer_ft.step()
                                self.scheduler_ft.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    epoch_loss = running_loss / len(self.dataloader_dict[phase].dataset)
                    epoch_acc = running_corrects.double() / len(self.dataloader_dict[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                    logging.info(('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)))
                    writer.writerow({'epoch': epoch, 'phase': phase, 'loss': epoch_loss, 'accuracy': epoch_acc.item()})
                    if phase == 'test':
                        test_data_store.append([epoch, phase, epoch_loss, epoch_acc.item()])

                    if phase == 'val':  # normaly we should take the min. val loss
                        improved = (best_acc <= epoch_acc)

                        if improved:
                            best_acc = epoch_acc
                            best_model_wts = copy.deepcopy(
                                self.model_ft.state_dict())
                            best_epoch = epoch
                            not_improved_count = 0
                        else:
                            not_improved_count += 1

                        val_acc_history.append(epoch_acc)

                if not_improved_count >= self.early_stop:
                    logging.info((
                        "Validation performance didn\'t improve for {} epochs. Training stops. "\
                        "The Score is {}%. Best Epoch is {}".format(self.early_stop, best_acc, best_epoch)))
                    print(
                        "Validation performance didn\'t improve for {} epochs. Training stops. "\
                        "The Score is {}%. Best Epoch is {}".format(self.early_stop, best_acc, best_epoch))

                    break

                    ###---------------------------------------------------------------###

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            logging.info(('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)))
            print('Best val Acc: {:4f}'.format(best_acc))
            logging.info(('Best val Acc: {:4f}'.format(best_acc)))
            print('Test', test_data_store[best_epoch])
            logging.info(('Test', test_data_store[best_epoch]))
            self.model_ft.load_state_dict(best_model_wts)

            # plot the confusion matrix
            with torch.no_grad():
                self.model_ft.eval()
                prediction_loader = self.dataloader_dict['test']
                test_preds, test_labels = self.get_all_preds(self.model_ft, prediction_loader)
                preds_correct = test_preds.argmax(dim=1).eq(test_labels).sum().item()
                print('total correct:', preds_correct)
                logging.info(('total correct:', preds_correct))
                print('accuracy:', preds_correct / len(test_labels))
                logging.info(('accuracy:', preds_correct / len(test_labels)))
                y_true = test_labels.cpu().detach().numpy().astype(int)
                y_pred = test_preds.argmax(dim=1).cpu().detach().numpy().astype(int)


                cm =confusion_matrix(y_true,y_pred, labels = [b for b in range(0,len(self.train_tile_classes))])
                print(cm)
                logging.info((cm))
                sm = nn.Softmax(dim=1)
                #print('preds shape', sm(test_preds).cpu().detach().numpy().shape)
                y_scori = sm(test_preds).cpu().detach().numpy()
                print('roc_auc_score', roc_auc_score(y_true, y_scori, multi_class='ovr',average='macro'))
                logging.info(('roc_auc_score', roc_auc_score(y_true, y_scori, multi_class='ovr',average='macro')))
                lab_temp = [b for b in range(0,len(self.train_tile_classes))]
                print('Jaccard Score for each class {}'.format(jaccard_score(y_true, y_pred, average=None)))
                logging.info(('Jaccard Score for each class {}'.format(jaccard_score(y_true, y_pred, average=None))))
                print(classification_report(y_true, y_pred, labels=lab_temp))
                logging.info((classification_report(y_true, y_pred, labels=lab_temp)))
                #print('roc_auc Score for each class {}'.format(roc_auc_score(y_true, y_scori, average='None', multi_class='ovr')))
                # sklearn.metrics.precision_score
                # sklearn.metrics.recall_score
                # sklearn.metrics.roc_auc_score
                if self.mod:
                    from sklearn.metrics import ConfusionMatrixDisplay # ToDo Bugfix for HPC
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = self.train_tile_classes)#
                    disp.plot()
                    plt.savefig('./logs/{}_confusion_matrix.jpg'.format(self.datestr()))
                    plt.show()
                print('Warning the confusion matrix need validation')

            # Save the model
            ts = self.datestr()
            save_path = self.save_path + 'train_model_{}_'.format(self.model_name) + '{}'.format(self.model_id)\
                        + str(ts) + '.pth'
            torch.save(self.model_ft.state_dict(), save_path)
            self.val_acc_history = val_acc_history


    def set_parameter_requires_grad(self, ):
        ''' Freeze the layers '''
        if self.feature_extract:
            for param in self.model_ft.parameters():
                param.requires_grad = False

    def initialize_model(self, ):
        ''' Here intialize the layers new, we want to train'''
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None

        if self.model_name == "resnet":
            """ Resnet18
            """
            self.model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "resnet50":
            """ Resnet50
            """
            self.model_ft = models.resnet50(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            self.model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            self.model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.classifier[6].in_features
            self.model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            self.model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            self.model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model_ft.num_classes = self.num_classes
            self.input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            self.model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            num_ftrs = self.model_ft.classifier.in_features
            self.model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 224

        elif self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad()
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            self.model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            self.input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

    def initialize_optimizer(self, ):
        ''' ToDo add more optimizer maybe parameter options'''
        if self.optimizer_name == 'SDG':
            self.optimizer_ft = optim.SGD(self.params_to_update, lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == 'ADAM':
            self.optimizer_ft = optim.Adam(self.params_to_update, lr=self.learning_rate, weight_decay=0.0001)
        else:
            self.optimizer_ft = optim.Adam(self.params_to_update, lr=self.learning_rate, weight_decay=0.0001)  # default

    def initialize_criterion(self, ):
        ''' Init the CL loss '''
        if self.criterion_name == 'CEL':
            self.criterion_ft = nn.CrossEntropyLoss()
        else:
            self.criterion_ft = nn.CrossEntropyLoss()

    def initialize_dataloader(self, ):
        ''' Init the dataloader '''
        # Data augmentation and normalization for training
        # Just normalization for validation
        if not self.reload:
            self.modi = self.train_val_test
        else:
            self.modi = ['ana']

        if not self.get_os():
            num_w = 4
        else:
            num_w = 0

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #transforms.Grayscale(num_output_channels=3)
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #transforms.Grayscale(num_output_channels=3)
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #transforms.Grayscale(num_output_channels=3)
            ]),
            'ana': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                #transforms.Grayscale(num_output_channels=3)
            ])
        }
        self.data_transforms = data_transforms
        print("Initializing Datasets and Dataloaders...")
        if self.img_data_dict:
            self.dataloader_dict = {
                x: torch.utils.data.DataLoader(
                    FromDict2Dataset(self.img_data_dict, phase=x, class_names=self.class_names,
                                     transform=data_transforms[x]),
                    batch_size=self.batch_size, shuffle=self.data_load_shuffle,
                    num_workers=num_w) for x in self.modi}

        else:
            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in
                              self.modi}
            # Create training and validation dataloaders
            self.dataloader_dict = {
                x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=self.data_load_shuffle,
                                               num_workers=num_w)
                for x in
                self.modi}


    def initialize_scheduler(self, ):
        ''' init the learning rate scheduler '''
        if self.scheduler_name == 'None':
            self.scheduler_ft = lr_scheduler.StepLR(self.optimizer_ft, step_size=self.lr_step_size, gamma=self.gamma)
        else:
            self.scheduler_ft = lr_scheduler.StepLR(self.optimizer_ft, step_size=self.lr_step_size, gamma=self.gamma)

    def select_retrained_children(self, ):
        ''' Select layers for retrain'''
        max_Children = int(len([child for child in self.model_ft.children()]))
        #print(max_Children)
        ct = max_Children
        for child in self.model_ft.children():
            ct -= 1
            if ct < self.num_train_layers:
                for param in child.parameters():
                    param.requires_grad = True

    def update_params(self, ):
        ''' Select params to update '''
        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = self.model_ft.parameters()
        #print("Params to learn:")
        if self.feature_extract:
            self.params_to_update = []
            for name, param in self.model_ft.named_parameters():
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

    def print_cuda(self):
        ''' Print out GPU Details and Cuda Version '''
        print('__Python VERSION:', sys.version)
        logging.info(('__Python VERSION:', sys.version))
        print('__pyTorch VERSION:', torch.__version__)
        logging.info(('__pyTorch VERSION:', torch.__version__))
        print('__CUDA VERSION:', torch.version.cuda)
        logging.info(('__pyTorch VERSION:', torch.__version__))
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        logging.info(('__CUDNN VERSION:', torch.backends.cudnn.version()))
        print('__Number CUDA Devices:', torch.cuda.device_count())
        logging.info(('__Number CUDA Devices:', torch.cuda.device_count()))
        print('__Devices:')
        logging.info('__Devices:')
        print('current session {} current user {} time {}'.format(self.session_id, getpass.getuser(),self.datestr()))
        logging.info(('current session {} current user {} time {}'.format(self.session_id, getpass.getuser(),self.datestr())))
        # print(__status__, '\n', __credits__)

    def get_os(self):
        ''' Print out operating system '''
        if os.name == 'posix':
            print('This System is Unix')
            mod = False
        elif os.name == 'nt':
            print('This System is Windows')
            mod = True
        return mod

    def imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    def visualize_model(self, ):
        ''' rework for script and dont use for hpc '''
        was_training = self.model_ft.training
        self.model_ft.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader_dict['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model_ft(inputs)
                _, preds = torch.max(outputs, 1)

                out = torchvision.utils.make_grid(inputs)
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(self.num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_names[preds[j]]))
                    self.imshow(inputs.cpu().data[j])


                    if images_so_far == self.num_images:
                        self.model_ft.train(mode=was_training)
                        return
            self.model_ft.train(mode=was_training)


    def get_that_tiles(self, ):
        '''Create tile form Fotos in Folders or just cut image and write them into dict in
            mod= 'ana',only create dict '''
        d = {}
        if self.dont_save_tiles != 'yes':
            os.mkdir(os.path.join(self.train_tile_path, str(self.tile_size)))
        for TVT in self.train_val_test:
            c = {}
            if self.dont_save_tiles != 'yes':
                os.mkdir(os.path.join(self.train_tile_path, str(self.tile_size), TVT))
            for TTC in self.train_tile_classes:
                Collector = []
                if self.dont_save_tiles != 'yes':
                    os.mkdir(os.path.join(self.train_tile_path, str(self.tile_size), TVT, TTC))
                images = glob.glob(os.path.join(self.train_tile_path, TVT, TTC, '*.tif'))
                for imgNr in images:
                    FileNamePure = os.path.basename(os.path.splitext(imgNr)[0])
                    #print(imgNr, FileNamePure)
                    imgA = image.imread(imgNr)
                    if self.resize:
                        imgA = self.resize_input(imgA)

                    CropImg = imgA[0:self.tile_size * (math.floor(imgA.shape[0] / self.tile_size)),
                              0:self.tile_size * (math.floor(imgA.shape[1] / self.tile_size)), :]
                    SplitArray = np.array(np.array_split(CropImg, CropImg.shape[0] / self.tile_size, axis=0))
                    SplitArray = np.array(np.array_split(SplitArray, SplitArray.shape[2] / self.tile_size, axis=2))
                    SplitArray = SplitArray.reshape(
                        int((CropImg.shape[0]) / self.tile_size * int(CropImg.shape[1] / self.tile_size)),
                        self.tile_size, self.tile_size, 3)
                    #print('type', type(SplitArray))
                    for tiles in range(len(SplitArray)):
                        if self.selection:
                            canidate = SplitArray[tiles]
                            if self.validate_input(canidate):
                                Collector.append(canidate)
                        else:
                            Collector.append(SplitArray[tiles])
                        if self.dont_save_tiles != 'yes':
                            FileNameNr = str(tiles) + FileNamePure
                            FileName = os.path.join(self.train_tile_path, str(self.tile_size), TVT, TTC,
                                                    FileNameNr + '.tif')
                            im = Image.fromarray(SplitArray[tiles])
                            im.save(FileName)
                c[TTC] = Collector
            d[TVT] = c
        self.img_data_dict = d

    def get_that_tiles(self, ):
        ''' NEW Create tile form Fotos in Folders or just cut image and write them into dict in
            mod= 'ana',only create dict '''
        d = {}
        if self.dont_save_tiles != 'yes':
            os.mkdir(os.path.join(self.train_tile_path, str(self.tile_size)))
        for TVT in self.train_val_test:
            c = {}
            if self.dont_save_tiles != 'yes':
                os.mkdir(os.path.join(self.train_tile_path, str(self.tile_size), TVT))
            for TTC in self.train_tile_classes:
                Collector = []
                if self.dont_save_tiles != 'yes':
                    os.mkdir(os.path.join(self.train_tile_path, str(self.tile_size), TVT, TTC))
                images = glob.glob(os.path.join(self.train_tile_path, TVT, TTC, '*.tif'))
                for imgNr in images:
                    FileNamePure = os.path.basename(os.path.splitext(imgNr)[0])
                    #print(imgNr, FileNamePure)
                    print('Processing imagefile: {}'.format(imgNr))
                    logging.info(('Processing imagefile: {}'.format(imgNr)))

                    SplitArray, FN = self.image2tile(imgNr, dir_out = None,  d=self.tile_size)
                    for tiles in range(len(SplitArray)):
                        if self.selection:
                            canidate = SplitArray[tiles]
                            #print(type(canidate))
                            #print(canidate)
                            if self.validate_input(canidate):#Todo
                                Collector.append(canidate)
                        else:
                            Collector.append(SplitArray[tiles])
                        if self.dont_save_tiles != 'yes':
                            FileNameNr = str(tiles) + FileNamePure
                            FileName = os.path.join(self.train_tile_path, str(self.tile_size), TVT, TTC,
                                                    FileNameNr + '.tif')
                            im = Image.fromarray(SplitArray[tiles])
                            im.save(FileName)
                c[TTC] = Collector
            d[TVT] = c
        self.img_data_dict = d

    def get_that_tiles_Arr_dict(self, imgNr):
        ''' Create a data dict for analyze (Dummy Label NL for No Label) '''
        c1 = {}
        d1 = {}
        Collector = []
        imgA = image.imread(imgNr)
        CropImg = imgA[0:self.tile_size * (math.floor(imgA.shape[0] / self.tile_size)),
                  0:self.tile_size * (math.floor(imgA.shape[1] / self.tile_size)), :]
        SplitArray = np.array(np.array_split(CropImg, CropImg.shape[0] / self.tile_size, axis=0))
        SplitArray = np.array(np.array_split(SplitArray, SplitArray.shape[2] / self.tile_size, axis=2))
        SplitArray = SplitArray.reshape(
            int((CropImg.shape[0]) / self.tile_size * int(CropImg.shape[1] / self.tile_size)),
            self.tile_size,
            self.tile_size, 3)
        for tiles in range(len(SplitArray)):
            if self.selection:
                canidate = SplitArray[tiles]
                if self.validate_input(canidate):
                    Collector.append(canidate)
            else:
                Collector.append(SplitArray[tiles])
        c1['NL'] = Collector
        d1['ana'] = c1
        #self.SplitArray = SplitArray
        self.img_data_dict = d1

    def get_that_tiles_Arr_dict(self, imgNr):
        '''NEW Create a data dict for analyze (Dummy Label NL for No Label) '''
        c1 = {}
        d1 = {}
        Collector = []
        SplitArray, FN = self.image2tile(imgNr, dir_out=None, d=self.tile_size)
        for tiles in range(len(SplitArray)):
            if self.selection:
                canidate = SplitArray[tiles]
                if self.validate_input(canidate):
                    Collector.append(canidate)
            else:
                Collector.append(SplitArray[tiles])
        c1['NL'] = Collector
        d1['ana'] = c1
        self.img_data_dict = d1

    def get_that_tiles_Arr_dict_new(self, imgNr):
        '''NEW Create a data dict for analyze (Dummy Label NL for No Label) '''
        c1 = {}
        d1 = {}
        Collector = []
        SplitArray, FN = self.image2tile(imgNr, dir_out=None, d=self.tile_size)
        print('Processing imagefile: {}'.format(imgNr))
        logging.info(('Processing imagefile: {}'.format(imgNr)))
        for tiles in range(len(SplitArray)):
            if self.selection:
                canidate = SplitArray[tiles]
                #if self.validate_input(canidate):
                Collector.append(canidate)
            else:
                Collector.append(SplitArray[tiles])
        c1['NL'] = Collector
        d1['ana'] = c1
        self.img_data_dict = d1

    def visualize_model_ana(self,):
        ''' rework for script and dont use for hpc '''
        was_training = self.model_ft.training
        self.model_ft.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader_dict['ana']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                # print(preds)

                out = torchvision.utils.make_grid(inputs)
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(self.num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_names[preds[j]]))
                    self.imshow(inputs.cpu().data[j])
                    print('Prediction:', self.class_names[preds[j]], 'Label:',
                          self.class_names[labels.cpu().data[j].item()])

                    if images_so_far == self.num_images:
                        self.model_ft.train(mode=was_training)
                        return
            self.model_ft.train(mode=was_training)

    @staticmethod
    def label_pic_based_on_calc_score(pred_decode, labs):
        ''' function to calc the labelsscores and select the mostlikely label '''
        l = [sum(np.array(list(map(lambda x: x == labs[i], pred_decode))).astype('int32')) / len(pred_decode)
             for i in range(0, len(labs))]
        print(labs[np.argmax(l)], l)
        return labs[np.argmax(l)], l

    def analyse_onePic_dict(self,):
        ''' function to analyze one full pic. returns the label for each tile, the labelscores and label '''
        self.model_ft.eval()
        predictions_decoded = []

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader_dict['ana']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                out = torchvision.utils.make_grid(inputs)
                for j in range(inputs.size()[0]):

                    predictions_decoded.append(self.class_names[preds[j]])
        labs = self.ignore_placeholder_label(self.class_names)
        pic_label, label_scores = self.label_pic_based_on_calc_score(predictions_decoded, labs)
        return predictions_decoded, pic_label, label_scores

    def analyse_onePic_dict_new(self,name):
        ''' function to analyze one full pic. returns the label for each tile, the labelscores and label '''
        self.model_ft.eval()
        predictions_decoded = []
        imgs = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloader_dict['ana']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #print('inputshape', inputs.shape)
                if self.validate_input(inputs.squeeze(0).cpu().numpy()) or not self.patho_cut_off: #ToDo
                    outputs = self.model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    out = torchvision.utils.make_grid(inputs)
                    for j in range(inputs.size()[0]):
                        predictions_decoded.append(self.class_names[preds[j]])
                    temp_img = self.color_img_by_label(inputs, preds.item())
                    imgs.append(torch.tensor(temp_img))
                else:
                    temp_img = self.color_img_by_label(inputs, 5)#Todo preds
                    imgs.append(torch.tensor(temp_img))
                    #print('B',torch.tensor(temp_img).shape)
                    #self.reject_counter += 1
        self.tile2image(list_of_tiles = imgs, filename = '{}'.format(name), dir_out = self.save_colored_dir)
        labs = self.ignore_placeholder_label(self.class_names)
        pic_label, label_scores = self.label_pic_based_on_calc_score(predictions_decoded, labs)
        return predictions_decoded, pic_label, label_scores

    def ignore_placeholder_label(self, class_names):
        ''' ingnore no label data procedure '''
        noNLlabels = []
        for i in class_names:
            if i == 'NL':
                pass
            else:
                noNLlabels.append(i)
        return noNLlabels

    def datestr(self, ):
        ''' get the date for naming of files '''
        temp_time ='{:04}_{:02}_{:02}__{:02}_{:02}_{:02}'.format(time.gmtime().tm_year, time.gmtime().tm_mon, time.gmtime().tm_mday,  time.gmtime().tm_hour, time.gmtime().tm_min, time.gmtime().tm_sec)
        return temp_time


    def create_field_names(self):
        ''' creating of files names '''
        pre_list = ['filename', 'pred_label']
        for lab in self.train_tile_classes:
            ent = lab + '_score'
            pre_list.append(ent)
        return pre_list

    def create_write_dict(self, pre_list, label_scores, pic_label, f_name):
        ''' write the label score dict for creation of the reports '''
        wdict = {}
        wdict[pre_list[0]] = f_name
        wdict[pre_list[1]] = pic_label
        for j in range(len(label_scores)):
            wdict[pre_list[j + 2]] = label_scores[j]
        return wdict

    def inference_folder(self):
        ''' analyze a folder of pictures '''
        self.reload_model()
        file_list = []
        with open('results/result_{}_{}_{}.csv'.format(self.session_id, self.result_file_name, self.datestr()), 'w', newline='') as csvfile:
            fieldnames = self.create_field_names()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

            writer.writeheader()
            for path, subdirs, files in os.walk(self.folder_path):
                for name in files:
                    print(os.path.join(path, name))
                    file_list.append(os.path.join(path, name))
                    self.get_that_tiles_Arr_dict(imgNr=os.path.join(path, name))
                    self.initialize_dataloader()
                    predictions_decoded, pic_label, label_scores = self.analyse_onePic_dict()
                    f_name = os.path.join(path, name)
                    writer.writerow(self.create_write_dict(fieldnames, label_scores, pic_label, f_name))
                    print(predictions_decoded, label_scores)
        return None

    def inference_folder_new(self):
        ''' analyze a folder of pictures '''
        self.reload_model()
        file_list = []
        with open('results/result_{}_{}_{}.csv'.format(self.session_id, self.result_file_name, self.datestr()), 'w', newline='') as csvfile:
            fieldnames = self.create_field_names()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            #print(self.folder_path, '#'*17)
            writer.writeheader()
            for path, subdirs, files in os.walk(self.folder_path):
                for name in files:
                    print(os.path.join(path, name))
                    file_list.append(os.path.join(path, name))
                    self.get_that_tiles_Arr_dict_new(imgNr=os.path.join(path, name))
                    self.initialize_dataloader()
                    predictions_decoded, pic_label, label_scores = self.analyse_onePic_dict_new(name)
                    f_name = os.path.join(path, name)
                    writer.writerow(self.create_write_dict(fieldnames, label_scores, pic_label, f_name))
        return None

    def resize_input(self, imgA):
        i_shape = imgA.shape
        x = i_shape[0]
        y = i_shape[1]
        mp_x = int(1/2 * (x-self.resize_shape[0]))
        mp_y = int(1/2 * (y - self.resize_shape[1]))
        imgA = imgA[mp_x:self.resize_shape[0] + mp_x, mp_y:self.resize_shape[1] + mp_y, :]
        return imgA


    def validate_input(self,canidate ):
        if self.reload: #for color_public
            canidate = canidate.transpose((2, 1, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            canidate = std * canidate + mean
            canidate = np.clip(canidate, 0, 1)
            canidate = canidate*255
            #print(canidate)
        else:
            canidate = canidate.transpose((2, 1, 0))
            canidate = canidate
        tissue = (canidate < self.pixel_cutoff)
        tissue_percentage = np.sum(tissue) / np.prod(tissue.shape)
        if not tissue_percentage > self.tissue_per:
            pass
        else:
            pass
        return (tissue_percentage > self.tissue_per)

    @torch.no_grad()
    def get_all_preds(self, model, loader):
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        for batch in loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = model(images)
            all_preds = torch.cat(
                (all_preds.to(self.device), preds.to(self.device))
                , dim=0
            )
            all_labels = torch.cat(
                (all_labels.to(self.device), labels.to(self.device))
                , dim=0
            )
        return all_preds, all_labels

##########################################################################################
    def color_img_by_label(self, img, label=None):
        img = img.clone().detach().requires_grad_(False).squeeze(0)
        r, g, b = torch.tensor_split(img, 3, dim=0)
        z = torch.zeros_like(r)
        if self.label_coloring:
            pos = [(z,g,b), (r,z,b), (r,g,z), (r,z,z), (z,g,z), (z,z,b)]
        else:
            pos = [(r, g, b), (r, g, b), (r, g, b), (r, g, b), (r, g, b), (r, g, b)]
        i = label
        img_new = torch.cat(pos[i], dim=0).cpu().numpy()
        return img_new


    def image2tile(self, path, dir_out, d=224):
        filename = re.split('\\\\', path)[-1]
        name, ext = os.path.splitext(filename)
        img = Image.open(os.path.join(path))
        if self.one_color_channel:
            print('Img Shape', np.array(img).shape)
            temp_tensor = torch.tensor(np.array(img), dtype=torch.uint8)
            temp3 = torch.cat([temp_tensor.unsqueeze(2),temp_tensor.unsqueeze(2),temp_tensor.unsqueeze(2)],dim=2)
            print('new im shape', temp3.shape)
            img = Image.fromarray(temp3.cpu().numpy())
        if self.resize:
            img = np.array(img)
            img = self.resize_input(img)
            img = Image.fromarray(img)
        w, h = img.size
        c = min(w,h)
        w ,h = c, c
        M = []
        grid = list(itertools.product(range(0, h - h % d, d), range(0, w - w % d, d)))
        for i, j in grid:
            box = (j, i, j + d, i + d)
            temp = torch.tensor(np.array(img.crop(box))).numpy()
            M.append(temp)
        return M, filename

    def tile2image(self, list_of_tiles, filename, dir_out):
        l = len(list_of_tiles)
        n = int(math.floor(np.sqrt(l)))
        C = [list_of_tiles[i:i+n] for i in range(0, l, n)]
        H = torch.cat([torch.cat([s for s in sub], dim=2) for sub in C], dim=1).cpu().numpy()
        H_new = H.transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        H_new = std * H_new + mean
        H_new = np.clip(H_new, 0, 1)
        H_new = (H_new*255).astype(np.uint8)
        im2 = Image.fromarray(H_new)
        out = os.path.join(dir_out, 'colored_{}_{}_{}'.format(self.session_id, self.datestr(),filename))
        im2.save(out)
        return H

