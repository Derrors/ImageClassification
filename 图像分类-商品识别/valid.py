from train import *

def model_valid(model_path, csv_path):
    model = dpn.dpn131(200).to(device)
    model.load_state_dict(torch.load(model_path))
    model_test(model, csv_path)

    return

if __name__ == '__main__':
    model_path = output_path + 'dpn1024_50.pth'
    csv_path = output_path + 'dpn_valid.csv'
    model_valid(model_path, csv_path)