rm -f packsim_core*
cd src
python3 setup.py build_ext --inplace --quiet
mv *.so ../