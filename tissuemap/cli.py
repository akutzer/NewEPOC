from omegaconf import OmegaConf, DictConfig
import argparse
from pathlib import Path
import os
from typing import Iterable, Optional
import shutil



NORMALIZATION_TEMPLATE_URL = "https://github.com/Avic3nna/STAMP/blob/main/resources/normalization_template.jpg?raw=true"
CTRANSPATH_WEIGHTS_URL = "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download"
DEFAULT_RESOURCES_DIR = Path(__file__).with_name("resources")
DEFAULT_CONFIG_FILE = Path("config.yaml")
TISSUEMAP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")


class ConfigurationError(Exception):
    pass


def _config_has_key(cfg: DictConfig, key: str):
    try:
        for k in key.split("."):
            cfg = cfg[k]
        if cfg is None:
            return False
    except KeyError:
        return False
    return True


def require_configs(cfg: DictConfig, keys: Iterable[str], prefix: Optional[str] = None):
    prefix = f"{prefix}." if prefix else ""
    keys = [f"{prefix}{k}" for k in keys]
    missing = [k for k in keys if not _config_has_key(cfg, k)]
    if len(missing) > 0:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")


def create_config_file(config_file: Optional[Path]):
    """Create a new config file at the specified path (by copying the default config file)."""
    config_file = config_file or DEFAULT_CONFIG_FILE
    # Locate original config file
    if not TISSUEMAP_FACTORY_SETTINGS.exists():
        raise ConfigurationError(f"Default tissueMAP config file not found at {TISSUEMAP_FACTORY_SETTINGS}")
    # Copy original config file
    shutil.copy(TISSUEMAP_FACTORY_SETTINGS, config_file)
    print(f"Created new config file at {config_file.absolute()}")


def resolve_config_file_path(config_file: Optional[Path]) -> Path:
    """Resolve the path to the config file, falling back to the default config file if not specified."""
    if config_file is None:
        if DEFAULT_CONFIG_FILE.exists():
            config_file = DEFAULT_CONFIG_FILE
        else:
            config_file = TISSUEMAP_FACTORY_SETTINGS
            print(f"Falling back to default tissueMAP config file because {DEFAULT_CONFIG_FILE.absolute()} does not exist")
            if not config_file.exists():
                raise ConfigurationError(f"Default tissueMAP config file not found at {config_file}")
    if not config_file.exists():
        raise ConfigurationError(f"Config file {Path(config_file).absolute()} not found (run `tissuemap init` to create the config file or use the `--config` flag to specify a different config file)")
    return config_file


def run_cli(args: argparse.Namespace):
    # Handle init command
    if args.command == "init":
        create_config_file(args.config)
        return

    # Load YAML configuration
    config_file = resolve_config_file_path(args.config)
    cfg = OmegaConf.load(config_file)

    # Set environment variables
    if "TISSUEMAP_RESOURCES_DIR" not in os.environ:
        os.environ["TISSUEMAP_RESOURCES_DIR"] = str(DEFAULT_RESOURCES_DIR)
    
    match args.command:
        case "init":
            return # this is handled above
        
        case "setup":
            # Download normalization template
            normalization_template_path = Path(cfg.preprocessing.normalization_template)
            normalization_template_path.parent.mkdir(parents=True, exist_ok=True)
            if normalization_template_path.exists():
                print(f"Skipping download, normalization template already exists at {normalization_template_path}")
            else:
                print(f"Downloading normalization template to {normalization_template_path}")
                import requests
                r = requests.get(NORMALIZATION_TEMPLATE_URL)
                with normalization_template_path.open("wb") as f:
                    f.write(r.content)
            # Download feature extractor model
            feat_extractor = cfg.preprocessing.feat_extractor
            if feat_extractor == 'ctp':
                model_path = f"{os.environ['TISSUEMAP_RESOURCES_DIR']}/ctranspath.pth"
            elif feat_extractor == 'uni':
                model_path = f"{os.environ['TISSUEMAP_RESOURCES_DIR']}/uni"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path.exists():
                print(f"Skipping download, feature extractor model already exists at {model_path}")
            else:
                if feat_extractor == 'ctp':
                    print(f"Downloading CTransPath weights to {model_path}")
                    import gdown
                    gdown.download(CTRANSPATH_WEIGHTS_URL, str(model_path))
                elif feat_extractor == 'uni':
                    print(f"Downloading UNI weights")
                    from uni.get_encoder import get_encoder
                    get_encoder(enc_name='uni', checkpoint='pytorch_model.bin', assets_dir=model_path)

        case "config":
            print(OmegaConf.to_yaml(cfg, resolve=True))

        case "train":
            # raise NotImplementedError()
            require_configs(
                cfg,
                ["train_dir", "valid_dir", "output_dir", "backbone", "batch_size", "binary", "ignore_categories"],
                prefix="training"
            )
            c = cfg.training
            if c.backbone == 'ctp':
                model_path = f"{os.environ['TISSUEMAP_RESOURCES_DIR']}/ctranspath.pth"
            elif c.backbone == 'uni':
                model_path = f"{os.environ['TISSUEMAP_RESOURCES_DIR']}/uni"
            else:
                model_path = c.backbone
            
            from .classifier.train import train
            model = train(
                train_dir=Path(c.train_dir),
                valid_dir=Path(c.valid_dir),
                save_dir=Path(c.output_dir),
                backbone=model_path,
                batch_size=c.batch_size,
                binary=c.binary,
                ignore_categories=c.ignore_categories,
                cores=c.cores  if 'cores' in c else 8
            )

        case "preprocess":
            require_configs(
                cfg,
                ["output_dir", "wsi_dir", "classifier_path", "cache_dir", "microns", "cores", "norm", "del_slide", "only_feature_extraction", "device", "normalization_template", "feat_extractor"],
                prefix="preprocessing"
            )
            c = cfg.preprocessing
            # Some checks
            if c.norm and not Path(c.normalization_template).exists():
                raise ConfigurationError(f"Normalization template {c.normalization_template} does not exist, please run `tissueMAP setup` to download it.")
            if c.feat_extractor == 'ctp':
                model_path = f"{os.environ['TISSUEMAP_RESOURCES_DIR']}/ctranspath.pth"
            elif c.feat_extractor == 'uni':
                model_path = f"{os.environ['TISSUEMAP_RESOURCES_DIR']}/uni"
            if not Path(model_path).exists():
                raise ConfigurationError(f"Feature extractor model {model_path} does not exist, please run `tissueMAP setup` to download it.")
            from .features.wsi_norm import preprocess
            preprocess(
                output_dir=Path(c.output_dir),
                wsi_dir=Path(c.wsi_dir),
                extractor_path=model_path,
                classifier_path=Path(c.classifier_path),
                cache_dir=Path(c.cache_dir),
                cache=c.cache if 'cache' in c else True,
                norm=c.norm,
                normalization_template=Path(c.normalization_template),
                del_slide=c.del_slide,
                only_feature_extraction=c.only_feature_extraction,
                keep_dir_structure=c.keep_dir_structure if 'keep_dir_structure' in c else False,
                cores=c.cores,
                target_microns=c.microns,
                patch_size=c.patch_size if 'patch_size' in c else 224,
                batch_size = c.batch_size if 'batch_size' in c else 64,
                device=c.device,
                feat_extractor=c.feat_extractor
            )

        case "visualize":
            raise NotImplementedError()
            
        case _:
            raise ConfigurationError(f"Unknown command {args.command}")



def main() -> None:
    parser = argparse.ArgumentParser(prog="tissueMAP", description="tissueMAP: tissue embedding visualizer using UMAP")
    parser.add_argument("--config", "-c", type=Path, default=None, help=f"Path to config file (if unspecified, defaults to {DEFAULT_CONFIG_FILE.absolute()} or the default tissueMAP config file shipped with the package if {DEFAULT_CONFIG_FILE.absolute()} does not exist)")

    commands = parser.add_subparsers(dest="command")
    commands.add_parser("init", help="Create a new tissueMAP configuration file at the path specified by --config")
    commands.add_parser("setup", help="Download required resources")
    commands.add_parser("config", help="Print the loaded configuration")

    commands.add_parser("train", help="Train a tissue classifier")
    commands.add_parser("preprocess", help="Preprocess whole-slide images into feature vectors")
    commands.add_parser("visualize", help="Visualize the feature vectors")

    args = parser.parse_args()

    # If no command is given, print help and exit
    if args.command is None:
        parser.print_help()
        exit(1)

    # Run the CLI
    try:
        run_cli(args)
    except ConfigurationError as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    print("AAA")
    main()
