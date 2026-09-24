

def get_model(name, **kwargs):

    if name == "swin_t":
        num_features = kwargs.get("num_features", 512)
        use_checkpoint = kwargs.get("use_checkpoint", False)
        from .swin import SwinTransformer
        return SwinTransformer(num_classes=num_features, use_checkpoint=use_checkpoint)


    else:
        raise ValueError()
