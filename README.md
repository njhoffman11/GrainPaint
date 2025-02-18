# GrainPaint
The dataset used to train GrainPaint is available at https://zenodo.org/record/8241535
# Installation
1. Clone the repository
2. Follow these instructions to install pytorch https://pytorch.org/get-started/locally/
3. Run `pip install -r requirements.txt` to install the rest of the dependencies.
# Training
We provide the modeles we used in our paper in the model_chekpoints folder, so training a model is not necessary if you want to use one of our pre-trained models. 
A model can be trained using train_ddm_diffusers32.ipynb
A batch size of 8 fits in 24GB of memory, if you have less you should reduce batch_size.
# Generation
Microstructures can be generated using generation_32_aniso.py. 
ddm_load_path stores the path to the pre-trained model to be loaded. 
ddm32_big_250.ckpt is the model trained on isotropic grain structures, ddm32_big_250_aniso.ckpt is the model trained on anisotropic grain structures.
tile_gen5.py creates a checkerboard-like generation plan and is suggested for isotropic microstructures, tile_gen6.py creates a center-out generation plan and is suggested for anisotropic microstructures.
To switch tile_gen scripts, change the import in generation_32_aniso.py.
