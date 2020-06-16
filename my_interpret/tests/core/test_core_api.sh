#!/bin/sh

clang_pp_bin=clang++
g_pp_bin=g++
os_type=`uname`
script_path=`dirname "$0"`
root_path="$script_path/../.."

build_core=1
for arg in "$@"; do
   if [ "$arg" = "-nobuildcore" ]; then
      build_core=0
   fi
done

if [ $build_core -eq 1 ]; then
   echo "Building Core library..."
   /bin/sh "$root_path/build.sh" -32bit
else
   echo "Core library NOT being built"
fi

compile_all="\"$root_path/tests/core/TestCoreApi.cpp\" -I\"$root_path/tests/core\" -I\"$root_path/core/inc\" -std=c++11 -fpermissive -O3 -march=core2"

if [ "$os_type" = "Darwin" ]; then
   # reference on rpath & install_name: https://www.mikeash.com/pyblog/friday-qa-2009-11-06-linking-and-install-names.html
   # the -l<library> parameter for some reason adds a lib at the start and .dylib at the end

   compile_mac="$compile_all -L\"$root_path/staging\" -Wl,-rpath,@loader_path"

   echo "Compiling TestCoreApi with $clang_pp_bin for macOS release|x64"
   [ -d "$root_path/tmp/clang/intermediate/release/mac/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/intermediate/release/mac/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/clang/bin/release/mac/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/bin/release/mac/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$clang_pp_bin $compile_mac -m64 -DNDEBUG -l_ebmcore_mac_x64 -o \"$root_path/tmp/clang/bin/release/mac/x64/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/clang/intermediate/release/mac/x64/TestCoreApi/TestCoreApi_release_mac_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_mac_x64.dylib" "$root_path/tmp/clang/bin/release/mac/x64/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/clang/bin/release/mac/x64/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling TestCoreApi with $clang_pp_bin for macOS debug|x64"
   [ -d "$root_path/tmp/clang/intermediate/debug/mac/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/intermediate/debug/mac/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/clang/bin/debug/mac/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/bin/debug/mac/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$clang_pp_bin $compile_mac -m64 -l_ebmcore_mac_x64_debug -o \"$root_path/tmp/clang/bin/debug/mac/x64/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/clang/intermediate/debug/mac/x64/TestCoreApi/TestCoreApi_debug_mac_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_mac_x64_debug.dylib" "$root_path/tmp/clang/bin/debug/mac/x64/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/clang/bin/debug/mac/x64/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling TestCoreApi with $clang_pp_bin for macOS release|x86"
   [ -d "$root_path/tmp/clang/intermediate/release/mac/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/intermediate/release/mac/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/clang/bin/release/mac/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/bin/release/mac/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$clang_pp_bin $compile_mac -m32 -DNDEBUG -l_ebmcore_mac_x86 -o \"$root_path/tmp/clang/bin/release/mac/x86/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/clang/intermediate/release/mac/x86/TestCoreApi/TestCoreApi_release_mac_x86_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_mac_x86.dylib" "$root_path/tmp/clang/bin/release/mac/x86/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/clang/bin/release/mac/x86/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling TestCoreApi with $clang_pp_bin for macOS debug|x86"
   [ -d "$root_path/tmp/clang/intermediate/debug/mac/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/intermediate/debug/mac/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/clang/bin/debug/mac/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/clang/bin/debug/mac/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$clang_pp_bin $compile_mac -m32 -l_ebmcore_mac_x86_debug -o \"$root_path/tmp/clang/bin/debug/mac/x86/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/clang/intermediate/debug/mac/x86/TestCoreApi/TestCoreApi_debug_mac_x86_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_mac_x86_debug.dylib" "$root_path/tmp/clang/bin/debug/mac/x86/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/clang/bin/debug/mac/x86/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

elif [ "$os_type" = "Linux" ]; then
   # to cross compile for different architectures x86/x64, run the following command: sudo apt-get install g++-multilib
   # "readelf -d <lib_filename.so>" should show library rpath:    $ORIGIN/    OR    ${ORIGIN}/    for Linux so that the console app will find the core library in the same directory as the app: https://stackoverflow.com/questions/6288206/lookup-failure-when-linking-using-rpath-and-origin
   # the -l<library> parameter for some reason adds a lib at the start and .so at the end

   compile_linux="$compile_all -L\"$root_path/staging\" -Wl,-rpath-link,\"$root_path/staging\" -Wl,-rpath,'\$ORIGIN/'"

   echo "Compiling TestCoreApi with $g_pp_bin for Linux release|x64"
   [ -d "$root_path/tmp/gcc/intermediate/release/linux/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/intermediate/release/linux/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/gcc/bin/release/linux/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/bin/release/linux/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$g_pp_bin $compile_linux -m64 -DNDEBUG -l_ebmcore_linux_x64 -o \"$root_path/tmp/gcc/bin/release/linux/x64/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/gcc/intermediate/release/linux/x64/TestCoreApi/TestCoreApi_release_linux_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_linux_x64.so" "$root_path/tmp/gcc/bin/release/linux/x64/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/gcc/bin/release/linux/x64/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling TestCoreApi with $g_pp_bin for Linux debug|x64"
   [ -d "$root_path/tmp/gcc/intermediate/debug/linux/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/intermediate/debug/linux/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/gcc/bin/debug/linux/x64/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/bin/debug/linux/x64/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$g_pp_bin $compile_linux -m64 -l_ebmcore_linux_x64_debug -o \"$root_path/tmp/gcc/bin/debug/linux/x64/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/gcc/intermediate/debug/linux/x64/TestCoreApi/TestCoreApi_debug_linux_x64_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_linux_x64_debug.so" "$root_path/tmp/gcc/bin/debug/linux/x64/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/gcc/bin/debug/linux/x64/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling TestCoreApi with $g_pp_bin for Linux release|x86"
   [ -d "$root_path/tmp/gcc/intermediate/release/linux/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/intermediate/release/linux/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/gcc/bin/release/linux/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/bin/release/linux/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$g_pp_bin $compile_linux -m32 -DNDEBUG -l_ebmcore_linux_x86 -o \"$root_path/tmp/gcc/bin/release/linux/x86/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/gcc/intermediate/release/linux/x86/TestCoreApi/TestCoreApi_release_linux_x86_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_linux_x86.so" "$root_path/tmp/gcc/bin/release/linux/x86/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/gcc/bin/release/linux/x86/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi

   echo "Compiling TestCoreApi with $g_pp_bin for Linux debug|x86"
   [ -d "$root_path/tmp/gcc/intermediate/debug/linux/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/intermediate/debug/linux/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   [ -d "$root_path/tmp/gcc/bin/debug/linux/x86/TestCoreApi" ] || mkdir -p "$root_path/tmp/gcc/bin/debug/linux/x86/TestCoreApi"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   compile_command="$g_pp_bin $compile_linux -m32 -l_ebmcore_linux_x86_debug -o \"$root_path/tmp/gcc/bin/debug/linux/x86/TestCoreApi/test_core_api\" 2>&1"
   compile_out=`eval $compile_command`
   ret_code=$?
   echo -n "$compile_out"
   echo -n "$compile_out" > "$root_path/tmp/gcc/intermediate/debug/linux/x86/TestCoreApi/TestCoreApi_debug_linux_x86_build_log.txt"
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   cp "$root_path/staging/lib_ebmcore_linux_x86_debug.so" "$root_path/tmp/gcc/bin/debug/linux/x86/TestCoreApi/"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi
   "$root_path/tmp/gcc/bin/debug/linux/x86/TestCoreApi/test_core_api"
   ret_code=$?
   if [ $ret_code -ne 0 ]; then 
      exit $ret_code
   fi


   echo Passed All Tests
else
   echo "OS $os_type not recognized.  We support $clang_pp_bin on macOS and $g_pp_bin on Linux"
   exit 1
fi
