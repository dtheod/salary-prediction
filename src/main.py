import hydra
from omegaconf import DictConfig, OmegaConf


from clean_data import clean_data
from outlier_removal import outlier_removal


@hydra.main(config_path = "../config",
            config_name = "main")


def main(config:DictConfig):

    clean_data(config)
    outlier_removal(config)


if __name__ == "__main__":
    main()









