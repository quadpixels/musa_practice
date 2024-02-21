MUSA programming practices

Similar to CUDA practices but done on a Moore Threads device.

## Machine setup

* Personally I use `ASUS TUF Gaming B550M-Plus WIFI II`. It is also necessary that you applly the following BIOS settings:
  * `Resize BAR` should be enabled
  * `UEFI` boot mode should be used
* Install OS
  * Install [Lubuntu 20.04](https://cdimage.ubuntu.com/lubuntu/releases/20.04.5/release/)
    * Other releases might encounter error while compiling kernel module.
      * Lubuntu is preferred over Xubuntu b/c Xubuntu by default enables composition which results in a blank screen.

## Installing dependencies

1. Follow [this link](https://www.mthreads.com/pes/drivers/driver-info?productType=DESKTOP&productModel=DESKTOP_MTT_S80&osVersion=MTT_S80_Ubuntu) to download the Linux driver (the websites says S80 but it also works with S70)

2. Follow [this link](https://developer.mthreads.com/sdk/download/musa?equipment=&os=&driverVersion=&version=) to download MUSA toolkit 1.4.1 (might need to register an MThreads developer account)

After installation you should have `/usr/local/musa`.

## Contents

* `deviceProperties`: prints MUSA device info

```
$ LD_LIBRARY_PATH=/usr/local/musa/lib ./deviceProperties
Device name: MTT S70
Total Global Memory: 7427825664B = 7083 MiB
Shared Memory per Block: 28672B = 28 KiB
Regs Per Block: 262144
...
```

* `radixSort`: an implementation of ["Fast 4-way parallel radix sorting on GPUs"](https://web.archive.org/web/20221012085306/https://vgc.poly.edu/~csilva/papers/cgf.pdf)

```
$ LD_LIBRARY_PATH=/usr/local/musa/lib ./radixSort 
Sorting 10000000 elts: 181.992 ms elapsed, 54.9475 M elts/s
Passed!
```