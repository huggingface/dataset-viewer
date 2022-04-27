#!/bin/bash

set -ex

python3/test.py pages $pages >/dev/null
python3/test.py outlines $outlines >/dev/null
python3/test.py paragraphs $paragraphs >/dev/null

cd trec-car-tools-example/
mvn org.codehaus.mojo:exec-maven-plugin:1.5.0:java -Dexec.mainClass="edu.unh.cs.treccar_v2.read_data.ReadDataTest"  -Dexec.args="header ../$pages" >/dev/null
mvn org.codehaus.mojo:exec-maven-plugin:1.5.0:java -Dexec.mainClass="edu.unh.cs.treccar_v2.read_data.ReadDataTest"  -Dexec.args="pages ../$pages" >/dev/null
mvn org.codehaus.mojo:exec-maven-plugin:1.5.0:java -Dexec.mainClass="edu.unh.cs.treccar_v2.read_data.ReadDataTest"  -Dexec.args="outlines ../$outlines" >/dev/null
mvn org.codehaus.mojo:exec-maven-plugin:1.5.0:java -Dexec.mainClass="edu.unh.cs.treccar_v2.read_data.ReadDataTest"  -Dexec.args="paragraphs ../$paragraphs" >/dev/null
