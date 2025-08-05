The npme code requires the Intel C++ compiler and associated Math Kernel Library (MKL). 
As of npme v1.2, npme has been updated to support the new free Intel oneAPI compiler suite.

To install the latest Intel compiler:
  https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html

Once installed, npme should compile without modification. The npme source code is contained 
in the /src directory, and command-line applications are in /app.

To compile using 4 threads:
  > make -j4

After successful compilation, there are several test directories in /test that contain 
example input files and testing scripts. For example:
  > cd ./test/01_npme_laplaceDM
  > ./run.sh

#Compatibility Notes:

- Version 1.1 and later are compatible with the **Intel oneAPI compiler**.
- Earlier releases, including:
    - npme v1.1 - Journal submission
    - npme v1.0 - Initial public release
  used the older Intel Classic C++ Compiler and does not compile with newer toolchains.

See manual.pdf for additional details, usage examples, and test case descriptions.
