from parameter_holder import *

if __name__ == '__main__':
    arg_dict = create_arg_dict(reload=True, batch_size=1, data_load_shuffle=False,
                               session_id ='validation', normalize_on=0,label_coloring=True,one_color_channel=True)
    args = get_arguments(arg_dict)
    T1 = Trainer(args)
    T1.inference_folder_new()