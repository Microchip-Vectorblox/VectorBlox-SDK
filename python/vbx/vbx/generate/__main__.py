from .vnnx_flow import generate_vnnx
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--xml', help='OpenVINO I8 model description (.xml)', required=True)
    parser.add_argument('-f', '--samples-folder', help='provide folder of sample images to gather layer statistics',required=True)
    parser.add_argument('-sc', '--samples-count', type=int, help='provide max number of sample images to run')
    parser.add_argument('-i', '--image', help='provide test input image for model, must be correct dimensions')
    parser.add_argument('-c', '--size-conf', help='size configuration to build model for',
                        choices = ['V250','V500','V1000'], required=True)
    parser.add_argument('--cut', help='Cuts graph after OpenVINO node (use name specified by Netron)')
    parser.add_argument('-o','--output',help="Name of output file",required=True)
    parser.add_argument('-bc', '--bias-correction', help="apply bias correction to layers to improve accuracy",action='store_true')
    parser.add_argument('-kld', '--kl-divergence', help="use KL divergence to determine layer thresholds",action='store_true')
    #undocumented arguments
    parser.add_argument('--keep-temp',help=argparse.SUPPRESS,action='store_true')#Keep temporary files around
    args = parser.parse_args()

    generate_vnnx(args.xml,
                  args.size_conf,
                  keep_temp=args.keep_temp,
                  image=args.image,
                  samples_folder=args.samples_folder,
                  samples_count=args.samples_count,
                  output_filename=args.output,
                  cut_node=args.cut,
                  bias_correction=args.bias_correction,
                  kld=args.kl_divergence)

if __name__ == "__main__":
    main()
