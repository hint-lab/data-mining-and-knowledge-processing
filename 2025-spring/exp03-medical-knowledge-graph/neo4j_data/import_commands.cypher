// 导入节点
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (n:MedicalEntity {
    id: row.id,
    type: row.type,
    name: row.name
});

// 导入关系
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
MATCH (start:MedicalEntity {id: row.start_id})
MATCH (end:MedicalEntity {id: row.end_id})
CREATE (start)-[r:RELATED_TO]->(end);

// 查看所有节点
MATCH (n) RETURN n;

// 查看特定类型的节点
MATCH (n:MedicalEntity) WHERE n.type = 'Symptom' RETURN n;

// 查看特定节点的关系
MATCH (n:MedicalEntity {name: '糖尿病'})-[r]-(m) RETURN n, r, m;