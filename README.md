# Anti-spoofing
This repository corresponds to an adapted implementation of the paper "On the generalization of color texture-based face anti-spoofing" by authors Zinelabidine Boulkenafet, Jukka Komulainen, and Abdenour Hadid. It is a discipline test in a PhD course entitled Computer Vision at the Informatics Center of the Federal University of Pernambuco, Brazil.

The codes used are adapted or collected from the following repositories:

LPQ:

bibtex
Copiar código
@misc{local_phase_quantisation,
  author = {Absaravanan},
  title = {Local Phase Quantisation},
  year = {2023},
  howpublished = {\url{https://gist.github.com/absaravanan/a145f3b1a364d2a499bca79525b2667b}},
  note = {Script for Local Phase Quantisation algorithm in Python},
  url = {https://gist.github.com/absaravanan/a145f3b1a364d2a499bca79525b2667b}
}
BSIF and the 8 filters with size 7x7:

bibtex
Copiar código
@misc{domain-specific-BSIF-for-iris-recognition,
  author = {Adam Czajka and Daniel Moreira and Kevin W. Bowyer and Patrick J. Flynn},
  title = {Domain-Specific Human-Inspired Binarized Statistical Image Features for Iris Recognition},
  year = {2019},
  howpublished = {\url{https://github.com/CVRL/domain-specific-BSIF-for-iris-recognition}},
  note = {README, GPL-3.0 license},
  institution = {CVRL},
  url = {https://github.com/CVRL/domain-specific-BSIF-for-iris-recognition},
  version = {1.0}
}
MiniDataset:

bibtex
Copiar código
@misc{depedri2021face,
  author = {Kevin Depedri and Matteo Brugnera},
  title = {Face Spoofing Detection Using Colour Texture Analysis},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KevinDepedri/Face-Spoofing-Detection-Using-Colour-Texture-Analysis}},
  note = {Student project for the "Signal, Image and Video" course, Master in Artificial Intelligent Systems at the University of Trento, a.y. 2021-2022}
}
LCCD Dataset:

bibtex
Copiar código
@INPROCEEDINGS{8895208,
  author = {Timoshenko, Denis and Simonchik, Konstantin and Shutov, Vitaly and Zhelezneva, Polina and Grishkin, Valery},
  booktitle = {2019 Computer Science and Information Technologies (CSIT)},
  title = {Large Crowdcollected Facial Anti-Spoofing Dataset},
  year = {2019},
  pages = {123-126},
  keywords = {computer science;biometrics;datasets},
  doi = {10.1109/CSITechnol.2019.8895208}
}
Extractors based on LBP:

bibtex
Copiar código
@misc{scikit-image,
  author = {scikit-image contributors},
  title = {skimage.feature.local_binary_pattern},
  year = {2024},
  url = {https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern},
  note = {Accessed: 2024-06-10}
}
SIFT:

bibtex
Copiar código
@misc{opencv_sift,
  author = {OpenCV},
  title = {Introduction to SIFT (Scale-Invariant Feature Transform)},
  year = {2023},
  url = {https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html},
  note = {Accessed: 2024-06-10}
}
.



The code is structured as follows: Each folder has an exact name that reflects its contents. For the code part, there are two main folders: one contains the actual code, and the other applies this code.
