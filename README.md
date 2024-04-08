# Analysis of Galaxy Evolution

GalaxyEvolution is a python package that calculates the evolution in
stellar mass of galaxies embedded in dark matter haloes of mass M200
at redshift z. For the halo assembly history I recommend the use of the 
commah package (Correa et al. 2015c).

# Installing

To get started set up a python virtual environment. The steps are as follows:

Clone GalaxyEvolution

```git clone https://github.com/correac/GalaxyEvolution.git```

```cd GalaxyEvolution```

```conda create -n commah_env python=3.10```

Now activate the virtual environment.

```conda activate commah_env```

```conda install -c conda-forge cosmolopy```

```pip install commah```

```pip install matplotlib```

Due to irregularities with the latests updates of the cosmolopy 
package I recommend installing commah via conda.

If you do not have anaconda, install miniconda : 

https://docs.anaconda.com/free/miniconda/index.html

and test it by typing ```conda list```