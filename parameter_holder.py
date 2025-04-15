from public_modul import *
import argparse

def create_arg_dict(
                    data_dir='./data/',
                    file_path_train='./data/',
                    train_val_test=['train', 'val', 'test'],
                    train_tile_classes=['apo', 'aut', 'fer', 'hea', 'nec'],
                    class_names=['apo', 'aut', 'fer', 'hea', 'nec', 'NL'],
                    tile_size=224,
                    dont_save_tiles='yes',
                    model_name="resnet50",
                    batch_size=100,
                    num_epochs=50,
                    num_train_layers=3,
                    feature_extract=True,
                    use_pretrained=True,
                    optimizer_name='ADAM',
                    criterion_name='CEL',
                    scheduler_name=None,
                    img_data_dict=None,
                    reload=True,
                    input_size=None,
                    save_path='./saved_models/',
                    folder_path=r'C:\Users\Admin\Desktop\pawel\exp6',
                    reload_path='./saved_models/train_model_resnet50_pawel_resnet50_ex1_ex42023_01_06__07_33_07.pth',
                    result_file_name='exp6result',
                    model_id='pawel_resnet50_ex1_ex4',
                    learning_rate=0.0001,
                    pixel_cutoff = 256,
                    tissue_per = 0.3,
                    early_stop = 25,
                    gamma = 0.95,
                    lr_step_size= 5,
                    normalize_on = 0,
                    data_load_shuffle = True,
                    label_coloring = True,
                    save_colored_dir = r'C:\Users\Admin\Desktop\pawel\exp6result',
                    session_id = 'User001',
                    patho_cut_off = False,
                    one_color_channel = True,
                    ):
    #ToDo Hook for GUI
    arg_dict = {}
    arg_dict['data_dir'] = data_dir
    arg_dict['train_tile_path'] = file_path_train
    arg_dict['train_val_test'] = train_val_test
    arg_dict['train_tile_classes'] = train_tile_classes
    arg_dict['class_names'] = class_names
    arg_dict['tile_size'] = tile_size
    arg_dict['dont_save_tiles'] = dont_save_tiles
    arg_dict['model_name'] = model_name
    arg_dict['batch_size'] = batch_size
    arg_dict['num_epochs'] = num_epochs
    arg_dict['num_train_layers'] = num_train_layers
    arg_dict['feature_extract'] = feature_extract
    arg_dict['use_pretrained'] = use_pretrained
    arg_dict['optimizer_name'] = optimizer_name
    arg_dict['criterion_name'] = criterion_name
    arg_dict['scheduler_name'] = scheduler_name
    arg_dict['img_data_dict'] = img_data_dict
    arg_dict['reload'] = reload
    arg_dict['input_size'] = input_size
    arg_dict['reload_path'] = reload_path
    arg_dict['folder_path'] = folder_path
    arg_dict['save_path'] = save_path
    arg_dict['result_file_name'] = result_file_name
    arg_dict['model_id'] = model_id
    arg_dict['learning_rate'] = learning_rate
    arg_dict['pixel_cutoff'] = pixel_cutoff
    arg_dict['tissue_per'] = tissue_per
    arg_dict['early_stop'] = early_stop
    arg_dict['gamma'] = gamma #
    arg_dict['lr_step_size'] = lr_step_size
    arg_dict['normalize_on'] = normalize_on
    arg_dict['data_load_shuffle'] = data_load_shuffle
    arg_dict['label_coloring'] = label_coloring
    arg_dict['save_colored_dir'] = save_colored_dir
    arg_dict['session_id'] = session_id
    arg_dict['patho_cut_off'] = patho_cut_off
    arg_dict['one_color_channel'] = one_color_channel
    print(arg_dict)
    return arg_dict


def get_arguments(arg_dict):
    parser = argparse.ArgumentParser()
    for ent in arg_dict.keys():
        parser.add_argument('--{}'.format(ent), type=type(arg_dict[ent]), default=arg_dict[ent])
    args = parser.parse_args() # "" when running on JupyterHub
    print(args)
    return args

if __name__ == '__main__':
    task = 'Train'
    if task == 'Infer':
        arg_dict = create_arg_dict(reload=True,normalize_on=0,one_color_channel=True)
        args = get_arguments(arg_dict)
        T1 = Trainer(args)
        T1.inference_folder()
    elif task == 'Train':
        arg_dict = create_arg_dict(reload=False,normalize_on=0,one_color_channel=True)
        args = get_arguments(arg_dict)
        T2 = Trainer(args)
        T2.model_train()
    else:
        print('No valid task')