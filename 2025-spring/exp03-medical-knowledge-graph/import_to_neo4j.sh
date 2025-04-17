#!/bin/bash

# 确保目录存在
mkdir -p data/neo4j/import

# 复制 CSV 文件到导入目录
cp data/processed/neo4j/nodes.csv data/neo4j/import/
cp data/processed/neo4j/relationships.csv data/neo4j/import/

# 重启 Neo4j 容器以应用新的数据卷
docker stop m.daocloud.io/docker.io/neo4j
docker rm m.daocloud.io/docker.io/neo4j
./start_neo4j_server.sh

# 等待 Neo4j 启动
echo "等待 Neo4j 启动..."
sleep 10

# 导入数据
echo "开始导入数据..."
docker exec neo4j cypher-shell -u neo4j -p password "
// 导入节点
LOAD CSV WITH HEADERS FROM 'file:///import/nodes.csv' AS row
CREATE (n:MedicalEntity {
    id: row.id,
    type: row.type,
    name: row.name
});

// 导入关系
LOAD CSV WITH HEADERS FROM 'file:///import/relationships.csv' AS row
MATCH (start:MedicalEntity {id: row.start_id})
MATCH (end:MedicalEntity {id: row.end_id})
CREATE (start)-[r:RELATED_TO]->(end);
"

echo "数据导入完成！" 