# string-dist

Attempting to quantitative how similar two strings are by a distance metric calculated using GPUs. First we will try using Tensorflow and Python.

The basic idea is to take an string and one-hot encode it, treating each character position as a categorical feature. 

Suppose we wanted to compare the two strings: "add" and "aecd" and the alphabet consisted of [a, b, c, d, e], we could convert them to this one-hot format and it would look like:

"add"  = 10000000100001000000

"aecd" = 10000000010010000010

where we have padded the first entry, now it's easily to apply any distance metric to provide a quantitative measure of how far apart they are. Note you must one-hot encode and cannot convert the characters at each position into a naive numerical values for the same reason categorical variables are one-hot encoded. I.E. they have no defined ordinality (green is not greater then red and "a" is not greater than "b".

The Hamming distance between these two strings is:5. 

Why would we want to do this? Because now for two lists of strings size N and M both sharing an alphabet of P characters, this becomes a matrix math problem and we can take advantage of all the machinery around GPU-accelerated fast matrix operations.

We can rapidly compute the NxM matrix of pair-wise similarity distances using matrix operations in Tensorflow or Numba. If our corpus is sufficiently large we can take advantage of sparse matrices to lighten our memory load.

There are some good tutorials on this on the internet and most of them are usable almost out of the box; [LINK].
