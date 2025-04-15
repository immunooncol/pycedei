from parameter_holder import*

if __name__ == '__main__':
    arg_dict = create_arg_dict(reload=False,normalize_on=0,one_color_channel=True)
    args = get_arguments(arg_dict)
    T2 = Trainer(args)
    T2.model_train()