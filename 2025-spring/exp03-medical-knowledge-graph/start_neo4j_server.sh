# 拉取Neo4j官方镜像
docker pull m.daocloud.io/docker.io/neo4j:latest

# 运行Neo4j容器
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v $(pwd)/data/neo4j:/var/lib/neo4j \
    -v $(pwd)/data/neo4j/import:/var/lib/neo4j/import \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    m.daocloud.io/docker.io/neo4j

# 等待Neo4j容器启动
sleep 10


