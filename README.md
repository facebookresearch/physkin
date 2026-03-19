# `PhySkin`

A neural framework for learning bone-based physics-aware cloth deformation.

## Requirements

### Environment Setup

The installation requires conda since it carries some useful packages not available from pypi.

1. Create a new conda environment
```bash
conda create -n physkin python=3.12
conda activate physkin
```

2. Clone submodules
```bash
git submodule update --init --recursive
```

3. Install [PyTorch](https://pytorch.org/)

Follow the official PyTorch installation instructions for your platform and CUDA version.

4. Install remaining requirements
```bash
pip install -r requirements.txt
```

## Data Setup

PhySkin requires prepared mesh data in the following structure:

```
<PATH_TO_YOUR_DATA_ROOT>/
├── bodies/
│   └── <body_name>/
│       ├── <mesh_name>.obj          # Body mesh (OBJ format, triangulated)
│       └── <data_name>.npy          # Body animation data (NumPy arrays)
└── garments/
    └── <garment_type>/
        ├── <garment_name>/
        │   └── <mesh_name>.obj      # Garment mesh (OBJ format, triangulated)
        └── <garment_type>_pattern_embeddings.pt  # Pattern embeddings (PyTorch tensor)
```

### File Format Requirements

- **Body meshes**: OBJ format, triangulated meshes
- **Garment meshes**: OBJ format, triangulated meshes
- **Animation data**: NPY format (NumPy arrays) with vertex positions
- **Pattern embeddings**: PT format (PyTorch tensor files)
- **Bone weights**: Included in mesh data dictionaries (defined in your data loader)

You must provide your own body and garment mesh data.

## Configuration

1. Copy and customize configuration files in `config/`:

```bash
cd config
cp physkin_hyperbone.yaml my_experiment.yaml
```

2. Update paths in your config file:

Replace all instances of `<PATH_TO_YOUR_DATA_ROOT>` with your actual data directory path.

Key configuration sections:
- `DATA_ROOT` / `LOCAL_DATA_ROOT`: Root paths to your data
- `body.custom_obj_path`: Path to body mesh OBJ file
- `body.verts_data_path`: Path to body animation data
- `garment.root_dir`: Path to garment meshes directory
- `garment.pattern_embeddings_path`: Path to pattern embeddings file

3. Validate your setup:

```bash
python validate_setup.py
```

This will check that:
- All dependencies are installed
- Configuration files have been customized
- Data paths are accessible

## Running

### Training

```bash
python train.py
```

To use a custom config:

```bash
python train.py --config-name=my_experiment.yaml
```

To override specific config values:

```bash
python train.py train.device=cuda:1 seed=42
```

### Inference

```bash
python infer.py
```

## Validation

Before running training or inference for the first time, validate your setup:

```bash
python validate_setup.py
```

This checks for:
- Missing dependencies
- Uncustomized config placeholders
- Missing data directories

Fix any reported issues and re-run validation until it passes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
