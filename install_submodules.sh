mkdir submodules
cd submodules
git submodule add https://github.com/tsmbland/membranequant
cd membranequant
git checkout 18cf083
cd ..
git submodule add https://github.com/tsmbland/polaritymodel
cd polaritymodel
git checkout 40c66fa
cd ../..
