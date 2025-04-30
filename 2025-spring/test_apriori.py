import itertools
import time
 
filename = "data.csv"
 
min_support = 2
 
#读取数据集
with open(filename) as f:
    content = f.readlines()
 
content = [x.strip() for x in content]
print(content)
 
Transaction = []                  #保存事务列表
Frequent_items_value = {}         #保存所有频繁项集字典
 
#将数据集的内容添加到事物列表
for i in range(0,len(content)):
    Transaction.append(content[i].split())
 
#获得频繁一项集
def frequent_one_item(Transaction,min_support):
    candidate1 = {}
 
    for i in range(0,len(Transaction)):
        for j in range(0,len(Transaction[i])):
            if Transaction[i][j] not in candidate1:
                candidate1[Transaction[i][j]] = 1
            else:
                candidate1[Transaction[i][j]] += 1
 
    frequentitem1 = []                      #获得满足最小支持度的频繁一项集
    for value in candidate1:
        if candidate1[value] >= min_support:
            frequentitem1 = frequentitem1 + [[value]]
            Frequent_items_value[tuple(value)] = candidate1[value]
 
    return frequentitem1
 
values = frequent_one_item(Transaction,min_support)
print(values)
print(Frequent_items_value)
 
 
# 从事物中删除不频繁的一项集
Transaction1 = []
for i in range(0,len(Transaction)):
    list_val = []
    for j in range(0,len(Transaction[i])):
        if [Transaction[i][j]] in values:
            list_val.append(Transaction[i][j])
    Transaction1.append(list_val)
 
 
#Hash节点类定义
class Hash_node:
    def __init__(self):
        self.children = {}           #指向子节点的指针
        self.Leaf_status = True      #了解当前节点是否为叶子节点的状态
        self.bucket = {}             #在储存桶中包含项目集
 
#构造得到Hash树类
class HashTree:
    # class constructor
    def __init__(self, max_leaf_count, max_child_count):
        self.root = Hash_node()
        self.max_leaf_count = max_leaf_count
        self.max_child_count = max_child_count
        self.frequent_itemsets = []
 
    # 进行递归插入以生成hashtree
    def recursively_insert(self, node, itemset, index, count):
        if index == len(itemset):
            if itemset in node.bucket:
                node.bucket[itemset] += count
            else:
                node.bucket[itemset] = count
            return
 
        if node.Leaf_status:
            ##########begin##########
            #如果node是叶结点所进行的操作代码
            if itemset in node.bucket:
                node.bucket[itemset]+=count
            else:
                node.bucket[itemset]=count
                if len(node.bucket)==self.max_leaf_count:
                    如果储存桶容量增加
                for old_itemset, old_count in node.bucket.items():
                    hash_key = self.hash_function(old_itemset[index])  #对下一个索引做哈希
                    if hash_key not in node.children:
                        node.children[hash_key] = Hash_node()
                    self.recursively_insert(node.children[hash_key], old_itemset, index + 1, old_count)
                del node.bucket
                node.Leaf_status = False
 
            ##########end##########                             
            
        else:
            ##########begin##########
            #如果node不是是叶结点所进行的操作代码
            hash_key=self.hash_function(itemset[index])
            if hash_key not in node.children:
                node.children[hash_key]=Hash_node()
            self.recursively_insert(node.children[hash_key],itemset,index+1,count)
 
 
            ##########end##########
            
 
    def insert(self, itemset):
        itemset = tuple(itemset)
        self.recursively_insert(self.root, itemset, 0, 0)
 
    # 添加支持度到候选项集中. 遍历树并找到该项集所在的储存桶.
    def add_support(self, itemset):
        Transverse_HNode = self.root
        itemset = tuple(itemset)
        index = 0
        while True:
            if Transverse_HNode.Leaf_status:
                if itemset in Transverse_HNode.bucket:    #在此储存桶中找到项集
                    Transverse_HNode.bucket[itemset] += 1 #增加此项目集的计数
                break
            hash_key = self.hash_function(itemset[index])
            if hash_key in Transverse_HNode.children:
                Transverse_HNode = Transverse_HNode.children[hash_key]
            else:
                break
            index += 1
 
 
    # 基于hash的支持度计算
    def get_frequent_itemsets(self, node, support_count,frequent_itemsets):
        ##########begin##########
        #获取频繁项集函数定义
        if node.Leaf_status:
            for key, value in node.bucket.items():
                if value >= support_count: 
                    #如果满足支持数条件
                    frequent_itemsets.append(list(key))   
                     #将其添加到频繁项集中
                    Frequent_items_value[key] = value
         
        for child in node.children.values():
            self.get_frequent_itemsets(child, support_count,frequent_itemsets)
 
 
 
        
        ##########end##########
    # hash function for making HashTree
    def hash_function(self, val):
        return int(val) % self.max_child_count
 
#生成hashTree
def generate_hash_tree(candidate_itemsets, max_leaf_count, max_child_count):
    htree = HashTree(max_child_count, max_leaf_count)             #create instance of HashTree
    for itemset in candidate_itemsets:
        htree.insert(itemset)                                     #to insert itemset into Hashtree
    return htree
 
#to generate subsets of itemsets of size k
def generate_k_subsets(dataset, length):
    subsets = []
    for itemset in dataset:
        subsets.extend(map(list, itertools.combinations(itemset, length)))
    return subsets
 
def subset_generation(ck_data,l):
    return map(list,set(itertools.combinations(ck_data,l)))
 
 
# 候选生成
 
def apriori_generate(dataset,k):
    ck = []
    #join step
    lenlk = len(dataset)
    for i in range(lenlk):
        for j in range(i+1,lenlk):
            L1 = list(dataset[i])[:k - 2]
            L2 = list(dataset[j])[:k - 2]
            if L1 == L2:
                ck.append(sorted(list(set(dataset[i]) | set(dataset[j]))))
 
    #prune step
    final_ck = []
    for candidate in ck:
        all_subsets = list(subset_generation(set(candidate), k - 1))
        found = True
        for i in range(len(all_subsets)):
            value = list(sorted(all_subsets[i]))
            if value not in dataset:
                found = False
        if found == True:
            final_ck.append(candidate)
 
    return ck,final_ck
 
 
# 候选剪枝
 
def generateL(ck,min_support):
    support_ck = {}
    for val in Transaction1:
        for val1 in ck:
            value = set(val)
            value1 = set(val1)
 
            if value1.issubset(value):
                if tuple(val1) not in support_ck:
                    support_ck[tuple(val1)] = 1
                else:
                    support_ck[tuple(val1)] += 1
    frequent_item = []
    for item_set in support_ck:
        if support_ck[item_set] >= min_support:
            frequent_item.append(sorted(list(item_set)))
            Frequent_items_value[item_set] = support_ck[item_set]
 
    return frequent_item
 
# apriori算法主函数
def apriori(L1,min_support):
    k = 2;
    L = []
    L.append(0)
    L.append(L1)
    max_leaf_count = 6  #每个hash树节点的最大容量
    max_child_count = 6  #每个hash树节点的最大子节点数
 
    start = time.time()
    while(len(L[k-1])>0):
        ck,final_ck = apriori_generate(L[k-1],k)                 #生成候选项集
        # print("C%d" %(k))
        # print(final_ck)
        h_tree = generate_hash_tree(ck,max_leaf_count,max_child_count)       #生成hash树
        if (k > 2):
            while(len(L[k-1])>0):
                l = generateL(final_ck, min_support)
                L.append(l)
                # print("Frequent %d item" % (k))
                # print(l)
                k = k + 1
                ck, final_ck = apriori_generate(L[k - 1], k)
                # print("C%d" % (k))
                # print(final_ck)
            break
        k_subsets = generate_k_subsets(Transaction1,k)                  #生成事物子集
        for subset in k_subsets:
            h_tree.add_support(subset)                                  #像hash树的项集添加支持数
        lk = []
        h_tree.get_frequent_itemsets(h_tree.root,min_support,lk)                  #获取频繁项集
        # print("Frequent %d item" %(k))
        # print(lk)
        L.append(lk)
        k = k + 1
    end = time.time()
    return L,(end-start)
 
L_value,time_taken = apriori(values,min_support)
#print("final L_value")
#print(L_value)
print("All frequent itemsets with their support count:")
print(Frequent_items_value)