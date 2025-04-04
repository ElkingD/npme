The npme code requires the Intel C++ compiler with the associated 
Math Kernel Library (MKL).  If the Intel C++ compiler is already installed,
npme should compile without issue. The npme library is contained in /src
and the npme applications are contained in /app.


For example, to compile on 4 threads:
>make -j4

After sucessful compiliation, there are various test directories in /test with
testing scripts "run.sh".  For example,
>cd ./test/01_npme_laplaceDM
>./run.sh

See manual.pdf for additional examples and details.



