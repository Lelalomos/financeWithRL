wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

tar -xvzf ta-lib-0.4.0-src.tar.gz

rm ta-lib-0.4.0-src.tar.gz

cd ta-lib

if [[ `id -un` == "root" ]]
    then
        ./configure --build=aarch64-unknown-linux-gnu
        make
        make install
    else
        ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu
        make
        make install
fi

yes | pip install ta-lib==0.4.32