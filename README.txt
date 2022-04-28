Instructions to run code. I've included my datasets for all intermediate steps in the zip file.
So, you can run steps 2-4 in any order without needing to wait on step 3, which could take a while.

"data/synth_Dk2_1.npy" - synthetic cell expression data matrix
"data/synth_Xk2_1.npy" - synthetic cell latent matrix
"data/synth_Yk2_1.npy" - synthetic gene latent matrix

"data/synth_loss_eSVDk2_curved_set3" - Reconstruction loss of latent matrix under expectation maximization
"data/synth_XeSVDk2_curved_set3" - Reconstructed cell latent matrix
"data/synth_YeSVDk2_curved_set3" - Reconstructed gene latent matrix

1. Install the following Python packages:
    a) math
    b) numpy
    c) matplotlib
    d) scipy
    e) sklearn
    f) random
    g) Slingshot (https://github.com/mossjacob/pyslingshot)

2. Run prepare_synthetic_dataset.py
    a) This will generate a synthetic cell expression dataset using a curved gaussian parametrized by a dot product model

3. Run main.py
    a) This will run the eSVD algorithm to regenerate a latent representation of the synthetic dataset

4. Run create_figures.py
    a) This will generate all figures used in the paper. Note that it generates more figures than were used in the paper.
        Also, this code has been updated and the output is better than was reported in the paper. It's still a work in progress.

5. Run EM.py
    a) This will perform EM on "data/synth_XeSVDk2_curved_set3.npy" using specified priors from slides.