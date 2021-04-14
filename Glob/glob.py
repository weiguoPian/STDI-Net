import argparse

def p_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path_NYC", default='./data/bikeNYC.npy', type=str)
    parser.add_argument(
        "--data_embedding", default='./data/embedding.npy', type=str)
    parser.add_argument(
        "--data_path_hour_feature", default='./data/hour_glove.npy', type=str)
    parser.add_argument("--seq_len", default=3, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_epoches", default=100, type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--lr", default=0.001, type=float) #0.001
    # parser.add_argument("--th", default=20, type=int)
    # parser.add_argument("--lr_decay", default=0.98, type=float)
    # parser.add_argument("--weight_decay", default=5e-5, type=float)

    parser.add_argument("--NYC_Height", default=16, type=int)
    parser.add_argument("--NYC_Weight", default=8, type=int)
    parser.add_argument("--hour_dim", default=300, type=int)

    args = parser.parse_args()

    return args