from parameter_holder import *

if __name__ == '__main__':
    arg_dict = create_arg_dict(reload=True,normalize_on=0)
    args = get_arguments(arg_dict)
    T1 = Trainer(args)
    T1.inference_folder()
