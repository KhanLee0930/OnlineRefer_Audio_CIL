from .referformer import build
from .onlinerefer import build as build_online
from .OnlineRefer_Aduio import build as build_onlinerefer_Aduio
from .OnlineRefer_Sign import build as build_onlinerefer_Sign

def build_model(args):
    if args.sign_online:
        print('#############################################################')
        print('Build Sign Online Model')
        print('#############################################################')
        return build_onlinerefer_Sign(args)
    if args.audio_online:
        print('#############################################################')
        print('Build Audio Online Model')
        print('#############################################################')
        return build_onlinerefer_Aduio(args)
    if args.online:
        print('#############################################################')
        print('Build Base Online Model')
        print('#############################################################')
        return build_online(args)
    elif args.semi_online:
        raise Exception(f'Now we does not support semi-online mode.')
    else:
        return build(args)
