import argparse
import random

import inference
import train

from omegaconf import OmegaConf

# 허깅페이스에 모델을 저장하고 싶으시면 실행 전 터미널에
# huggingface-cli login 입력 후
# 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 토큰값을 입력해주세요.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 여기서 omegaconfig 파일 이름 설정하고 실행해주세요.
    parser.add_argument("--config", "-c", type=str, default="klue-roberta-large")
    parser.add_argument("--mode", "-m", required=True)

    args = parser.parse_args()
    conf = OmegaConf.load(f"./config/{args.config}.yaml")

    # 시드 설정을 해야될까요?
    # SEED = conf.utils.seed
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.use_deterministic_algorithms(True)

    # 터미널 실행 예시 : python main.py -mt -> train.py 실행
    #                python main.py -mi -> inference.py 실행

    print("실행 중인 config file: ", args.config)
    if args.mode == "train" or args.mode == "t":
        train.train(conf)

    elif args.mode == "inference" or args.mode == "i":
        if conf.path.load_model_path is None:
            print("로드할 모델의 경로를 입력해주세요.")
        else:
            inference.inference(conf)
    else:
        print("실행모드를 다시 입력해주세요.")
        print("train     : t,\ttrain")
        print("inference : i,\tinference")
