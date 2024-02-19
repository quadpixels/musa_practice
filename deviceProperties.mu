#include <stdio.h>

#include <map>
#include <string>
#include <vector>

#include <musa.h>
#include <musa_runtime_api.h>

#include <sys/stat.h>
#include <elf.h>

#if defined(__LP64__)
#define ElfW(type) Elf64_ ## type
#else
#define ElfW(type) Elf32_ ## type
#endif

// Build manually
// /usr/local/musa/bin/mcc deviceProperties.mu -lmusart -lmusa
//
// Run:
// LD_LIBRARY_PATH=/usr/local/musa/lib ./a.out

void LoadFile(const char* fn) {
  ElfW(Ehdr) elf_header;
  FILE* f = fopen(fn, "rb");

  if (!f) {
    printf("Could not open file.\n");
    return;
  }

  std::vector<std::string> strings;
  std::map<int, std::string> offset2strings;
  fread(&elf_header, sizeof(elf_header), 1, f);
  if (memcmp(elf_header.e_ident, ELFMAG, SELFMAG) == 0) {
    printf("%s is a valid ELF file.\n", fn);
    printf("  e_shnum: %d\n", elf_header.e_shnum);
    
    ElfW(Shdr) *sections = (ElfW(Shdr)*)malloc(elf_header.e_shnum * elf_header.e_shentsize);
    fseek(f, elf_header.e_shoff, SEEK_SET);
    fread(sections, elf_header.e_shentsize, elf_header.e_shnum, f);

    // Find string table
    for (int i=0; i<elf_header.e_shnum; i++) {
      ElfW(Shdr)* sh = &(sections[i]);
      printf("section[%d], sh_name=%d, sh_type=%d, sh_size=%lu\n", i, sh->sh_name, sh->sh_type, sh->sh_size);
      if (sh->sh_type == SHT_STRTAB) {
        printf("Loading string table..\n");
        fseek(f, sh->sh_offset, SEEK_SET);
        char* tmp = new char[sh->sh_size];
        fread(tmp, sizeof(char), sh->sh_size, f);
        std::string cur = "";
        int offset = 0;
        for (int i=0; i<sh->sh_size; i++) {
          const char c = tmp[i];
          if (c == 0) {
            strings.push_back(cur);
            offset2strings[offset] = cur;
            cur = "";
          } else {
            if (cur.empty()) { offset = i; }
            cur.push_back(c);
          }
        }
        printf("%zu strings:\n", strings.size());
        for (int i=0; i<strings.size(); i++) {
          const std::string& s = strings[i];
          printf("  %d. %s\n", i, s.c_str());
        }
      }
    }
    
    // Locate ".mt_fatbin" section
    for (int i=0; i<elf_header.e_shnum; i++) {
      ElfW(Shdr)* sh = &(sections[i]);
      int name = sh->sh_name;
      if (name != 0 && offset2strings[name] == ".mt_fatbin") {
        printf("Section %d is .mt_fatbin.\n", i);
        const int LEN = sh->sh_size + 1;
        char* fatbin = new char[LEN];
        memset(fatbin, 0x00, LEN);
        fseek(f, sh->sh_offset, SEEK_SET);
        fread(fatbin, sizeof(char), sh->sh_size, f);
        MUmodule module;
        MUresult res = muModuleLoadFatBinary(&module, fatbin);
        printf("Res=%d\n", res);
        
        MUdevice device;
        muDeviceGet(&device, 0);
        
        MUfunction func;
        res = muModuleGetFunction(&func, module, "Hello");
        printf("Res=%d\n", res);
        
        muLaunchKernel(func, 2, 1, 1, 2, 1, 1, 0, 0, nullptr, nullptr);
      }
    }
  }
}

int main() {
  struct musaDeviceProp prop;
  musaError_t err;
  err = musaGetDeviceProperties(&prop, 0);
  if (err != musaSuccess) {
    printf("An error occurred.\n");
    return 1;
  }
  printf("Device name: %s\n", prop.name);
  printf("Total Global Memory: %zuB = %d MiB\n", prop.totalGlobalMem, int(prop.totalGlobalMem/1024/1024));
  printf("Shared Memory per Block: %zuB = %d KiB\n", prop.sharedMemPerBlock, int(prop.sharedMemPerBlock/1024));
  printf("Regs Per Block: %d\n", prop.regsPerBlock);
  printf("Warp Size: %d\n", prop.warpSize);
  printf("Mem Pitch: %zu\n", prop.memPitch);
  printf("Max Thread per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Thread Dim: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Clock Rate: %d\n", prop.clockRate);
  printf("Total Const Mem: %zu\n", prop.totalConstMem);
  printf("Major and Minor: %d.%d\n", prop.major, prop.minor);
  printf("Texture Alignment: %zu\n", prop.textureAlignment);
  printf("Texture Pitch Alignment: %zu\n", prop.texturePitchAlignment);
  printf("Device Overlap: %d\n", prop.deviceOverlap);
  printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
  printf("Kernel Exec Timeout Enabled: %d\n", prop.kernelExecTimeoutEnabled);
  printf("Integrated: %d\n", prop.integrated);
  printf("Can Map Host Memory: %d\n", prop.canMapHostMemory);
  printf("Compute Mode: %d\n", prop.computeMode);
  printf("Max Texture 1D:        %d\n", prop.maxTexture1D);
  printf("Max Texture 1D Mipmap: %d\n", prop.maxTexture1DMipmap);
  printf("Max Texture 1D Linear: %d\n", prop.maxTexture1DLinear);
  printf("Max Texture 2D:        %d x %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
  printf("Max Texture 2D Mipmap: %d x %d\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]);
  printf("Max Texture 2D Linear: %d x %d\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1]);
  printf("Max Texture 3D:        %d x %d x %d\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
  printf("Max Texture 3D Alt:    %d x %d x %d\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]);
  printf("Max Texture Cubemap:   %d\n", prop.maxTextureCubemap);
  printf("Max Texutre 1D Layered: %d x %d\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);
  printf("Max Texutre 2D Layered: %d x %d x %d\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
  printf("Max Surface Cubemap:         %d\n", prop.maxSurfaceCubemap);
  printf("Max Surface Cubemap Layered: %d x %d\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);
  printf("Surface Alignment: %zu\n", prop.surfaceAlignment);
  printf("Concurrent Kernels: %d\n", prop.concurrentKernels);
  printf("ECC Enabled: %d\n", prop.ECCEnabled);
  printf("PCI Bus:Device:Domain: %d:%d.%d\n", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
  printf("TCC Driver: %d\n", prop.tccDriver);

  return 0;
}
