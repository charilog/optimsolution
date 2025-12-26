# optimSolution
# Build & Run (Debug)

---

## Windows 

### Prerequisites (Windows 11)
- Visual Studio 2022 (Community or Build Tools)  
  - Workload: Desktop development with C++  
  - Components: MSVC v143, Windows 10/11 SDK, CMake Tools (optional but recommended)  
- CMake (if not installed via Visual Studio): ensure it is available in your system PATH  
- (Optional) Ninja for faster builds  

### Configure / Build / Run (Debug)
```bat
cd /path/to/optimsolution
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
cd build
.\Debug\optimsolution jso rastrigin 30
```

---

## Linux

### Install
```bash
sudo apt update
sudo apt install -y build-essential cmake gdb
sudo apt install -y ninja-build
```

### Configure / Build / Run (Debug)
```bash
cd /path/to/optimsolution
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
cd build
./optimsolution jso rastrigin 30
```

