GENCCODE_ASSEMBLY_TYPE=-a gcc
SO=so
SOBJ=so
A=a
LIBPREFIX=lib
LIB_EXT_ORDER=.70.1
COMPILE=/usr/bin/cc -D_REENTRANT  -DU_HAVE_ELF_H=1 -DU_HAVE_STRTOD_L=1 -DU_HAVE_XLOCALE_H=0  -DU_ATTRIBUTE_DEPRECATED= -O3  -Wno-undef -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -c
LIBFLAGS=-I../../../extern_icu/icu4c/source/common -I../../common -DPIC -fPIC
GENLIB=/usr/bin/cc -O3  -Wno-undef -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -Wl,-zorigin,-rpath,'$'ORIGIN -Wl,-rpath,/usr/local/lib -shared -Wl,-Bsymbolic
LDICUDTFLAGS=-nodefaultlibs -nostdlib
LD_SONAME=-Wl,-soname -Wl,
RPATH_FLAGS=
BIR_LDFLAGS=-Wl,-Bsymbolic
AR=ar
ARFLAGS=r
RANLIB=ranlib
INSTALL_CMD=/usr/bin/install -c
