# string-dist

Attempting to calculate string distance using GPUs (Hamming distance). First we will try using Tensorflow and python.

The basic idea is to take an string and one-hot encode it, treating each character position as a categorical feature. 

Suppose we wanted to compare the two strings: "add" and "aecd" and the alphabet consisted of [a, b, c, d, e], we could convert them to this one-hot format and it would look like:

"add"  = 10000000100001000000
"aecd" = 10000000010010000010

where we have padded the first entry, now it's easily to apply any distance metric to provide a quantitative measure of how far apart they are. Note you must one-hot encode and cannot convert the characters at each position into a naive numerical values for the same reason categorical variables are one-hot encoded. I.E. they have no defined ordinality (green is not greater then red and "a" is not greater than "b".

The Hamming distance between these two strings is "5". 

Why would we want to do this? Because now for two lists of strings size N and M both sharing an alphabet of P characters, this becomes a matrix math problem and we can take advantage of all the machinery around GPU-accelerated fast matrix operations.
