curl -c cookies http://localhost:8080/create_session/StellenboschFuji/4K-F2/DSCF1008.MOV
curl -b cookies http://localhost:8080/info
curl -b cookies -o frame1.jpeg "http://localhost:8080/frame?width=1280&height=720&format=jpeg"
curl -b cookies -o frame2.jpeg "http://localhost:8080/frame?width=1280&height=720&format=jpeg"
curl -b cookies -o frame3.jpeg "http://localhost:8080/frame?width=1280&height=720&format=jpeg"
curl -b cookies -o frame4.lz4 "http://localhost:8080/frame?width=1280&height=720&format=lz4"
curl -b cookies -o frame5.lz4 "http://localhost:8080/frame?width=1280&height=720&format=lz4"
curl -b cookies -o frame6.lz4 "http://localhost:8080/frame?width=1280&height=720&format=lz4"
