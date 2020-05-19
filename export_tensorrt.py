import torch
from os.path import join as pjoin
from models import get_model, get_lossfun
#from torch2trt import torch2trt
import argparse


def convert(args):
    model_name_F = args.arch_F
    model_F = get_model(model_name_F, True)  # concat and output
    model_F = torch.nn.DataParallel(model_F, device_ids=range(torch.cuda.device_count()))
    # Setup the map model
    if args.arch_map == 'map_conv':
        model_name_map = args.arch_map
        model_map = get_model(model_name_map, True)  # concat and output
        model_map = torch.nn.DataParallel(model_map, device_ids=range(torch.cuda.device_count()))

    print("Load training model: " + args.model_full_name)
    checkpoint = torch.load(pjoin(args.model_savepath, args.model_full_name))
    model_F.load_state_dict(checkpoint['model_F_state'])
    model_map.load_state_dict(checkpoint["model_map_state"])
    print('saving model ...')
    torch.save(model_map.state_dict(), pjoin(args.model_savepath, args.model_full_name)[:-4] + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--arch_F', nargs='?', type=str, default='fconv_ms',
                        help='Architecture for Fusion to use [\'fconv,fconv_in, fconv_ms etc\']')
    parser.add_argument('--arch_map', nargs='?', type=str, default='map_conv',
                        help='Architecture for confidence map to use [\'mask, map_conv etc\']')
    parser.add_argument('--model_savepath', nargs='?', type=str, default='./checkpoint/FCONV_MS',
                        help='Path for model saving [\'checkpoint etc\']')
    parser.add_argument('--model_full_name', nargs='?', type=str, default='',
                        help='The full name of the model to be tested.')

    args = parser.parse_args()
    convert(args)
