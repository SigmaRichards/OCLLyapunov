# OpenCL Implemented Lyapunov Fractals

This implementation of Lyapunov Fractals is designed to run on any OpenCL compatible device. Once compiled it will be available as the binary "./DO-THE-ROAR" (I know, very creative).


## Usage

General use will be like:
`./DO-THE-ROAR [options]`

Below are all the options. All options are optional.

 - `-l` lists the available OpenCL devices on your system. Once implemented you will be able to specify a specific device you want the program to run on.

 - `-w` and `-h` specifies the width and height (respectively) of the image you want rendered. This value should be an integer.

 - `-s` specifies the sequence of As and Bs (given as 0s and 1s respectively). Values should be sperated by a comma. Example: `-s 0,1`

 - `-a` and `-b` specifies the range to copute the $r$ values for A and B respectively. This should be given as a pair of float values separated by a comma. Example: `-a 3.0,4` will compute evenly separated $r$ values from $1$ to $4$.

 - `-x` specifies the starting $x_0$ value for each point. This value should be a single float. Example: `-x 0.500`. 

 - `-o` specifies the output name for the image. Example: `-o "out.png"`.

 - `-p` power term to increase contrast. Since the exponents are in the infinite domain, the need to be converted to 0-1 to be output as an image. This is done by computing $l = \exp{-|\lambda|}$. By default, most values are close to $1$ meaning there is little contrast. This parameter essentially computes $\hat{l} = l^p$ movnig more points further closer to 0. If you'd like to increase contrast in processing step you can increase this value to an integer. Example: `-p 3`.

 - `-n` number of iterations to calculate limit over. Since the actual lyupanov exponent needs to be calculated as the limit $n$ approaches infinity, we need to specify some stopping point. Example: `-n 10000`.

 - `-1` and `-2` specify the colours to use in the image creation process. `-1` specifies the color of stable attractors, while `-2` specifies the color of chaotic attractors. These values should be given as hex codes. Because of the way the exponents are calculated these will be the colors of the brightest pixels. Colors are RGB. Example: `-1#FF0000 -2 #0000FF`


## Explanation

Consider the logistic map:

$$x_{n+1} = rx_n(1-x_n)$$

And some binary sequence of As and Bs:

$$s = {A,B,A,A}$$

We can consider a new map:

$$x_{n+1} = r_{s_n}x_n(1-x_n)$$

With $r_{s_n}= r_A$ if $s_n=A$ or $r_{s_n} = r_B$ if $s_n==B$. We can evaluate the stability of the map for given values $r_A,r_B$ by calculating the lyapunov exponent:

$$\lambda = \lim_{n\rightarrow \infty} \frac{1}{n} \sum_{i=1}^{n} \log |f'(x_i)|$$

We can generate a fractal by considering the grid/matrix of $r_A,r_B$ pairs over a given range in $\mathcal{R}^2$ for some initial starting value $x_0$.




