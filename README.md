# GrainPaint
The dataset used to train GrainPaint is available at https://zenodo.org/record/8241535
# Installation
1. Clone the repository
2. Follow these instruction to install pytorch https://pytorch.org/get-started/locally/
3. Run `pip install -r requirements.txt` to install the rest of the dependencies.
# Training
A model can be trained using train_ddm_diffusers32.ipynb
A batch size of 8 fits in 24GB, if you have less you should reduce batch_size.
