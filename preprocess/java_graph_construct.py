import javalang
from javalang.ast import Node
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, DataCollatorWithPadding
from anytree import AnyNode
from tqdm import tqdm
from treelib import Tree
import re
import operator
from functools import reduce

'''
Child 0
Parent 1
NextSib 2
NextUse 3
NextToken 4
SplitChild 5
SplitParent 6
SplitNextSib 7
LoopNext
'''
AST_EDGE = 0
# Parent = 1
NextSib = 1
NextUse = 2
NextToken = 3
# SplitChild = 4
# SplitParent = 5
# SplitNextSib = 6
LoopNext = 4
ControlOut = 5
ConditionNext = 6

checkpoint = 'microsoft/codebert-base'
tokenize_token = '_<SplitNode>_'
ast_tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
javalang_special_tokens = ['CompilationUnit', 'Import', 'Documented', 'Declaration', 'TypeDeclaration', 'PackageDeclaration',
                           'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationDeclaration', 'Type',
                           'BasicType', 'ReferenceType', 'TypeArgument', 'TypeParameter', 'Annotation', 'ElementValuePair',
                           'ElementArrayValue', 'Member', 'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration',
                           'ConstantDeclaration', 'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration',
                           'VariableDeclarator', 'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement',
                           'WhileStatement', 'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement', 'ContinueStatement',
                           'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement', 'SwitchStatement',
                           'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause', 'CatchClauseParameter',
                           'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression', 'Assignment', 'TernaryExpression',
                           'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression', 'Primary', 'Literal', 'This',
                           'MemberReference', 'Invocation', 'ExplicitConstructorInvocation', 'SuperConstructorInvocation',
                           'MethodInvocation', 'SuperMethodInvocation', 'SuperMemberReference', 'ArraySelector', 'ClassReference',
                           'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator', 'InnerClassCreator', 'EnumBody',
                           'EnumConstantDeclaration', 'AnnotationMethod', 'Modifier', tokenize_token]
special_tokens_dict = {'additional_special_tokens': javalang_special_tokens}
num_added_toks = ast_tokenizer.add_special_tokens(special_tokens_dict)


def visiulize_tree(any_node):
    tree = Tree()

    def new_tree(node, parent=None):
        if node is None:
            return
        tree.create_node(node.token, node.id, parent=(
            None if not parent else parent.id))
        for child in node.children:
         #  print(child.token)
            new_tree(child, node)

    new_tree(any_node)

    tree.show()

# use javalang to generate ASTs and depth-first traverse to generate ast nodes corpus


def get_token(node):
    token = 'None'
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)


def parse_program(func):
    tokens = javalang.tokenizer.tokenize(func)
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()
    return tree


#  generate tree for AST Node
def create_tree(root, node, node_list, sub_id_list, leave_list, tokenizer, parent=None):
    id = len(node_list)
    node_list.append(node)
    token, children = get_token(node), get_child(node)

    if children == []:
        # print('this is a leaf:', token, id)
        leave_list.append(id)

    # Use roberta.tokenizer to generate subtokens
    # If a token can be divided into multiple(>1) subtokens, the first subtoken will be set as the previous node,
    # and the other subtokens will be set as its new children
    token = token.encode('utf-8', 'ignore').decode("utf-8")
    sub_token_list = tokenizer.tokenize(token)

    if len(sub_token_list) == 1 and sub_token_list[0] in javalang_special_tokens:
        pass
    # TODO convert token into lower
    else:
        sub_tokens = [tokenizer.tokenize(i.lower()) for i in re.sub(
            '([a-z0-9])([A-Z])', r'\1 \2', token).split()]
        for i in range(len(sub_tokens)):
            sub_tokens[i][0] = 'Ġ' + sub_tokens[i][0]
        sub_token_list = reduce(operator.add, sub_tokens)
        # print(sub_token_list)

    #  # TODO 叶子节点加上空白符号
    # if children is None or len(children) == 0:
    #     sub_token_list[0] = 'Ġ' + sub_token_list[0]

    if id == 0:
        # the root node is one of the tokenizer's special tokens
        root.token = sub_token_list[0]
        root.data = node
        # record the num of nodes for every children of root
        root_children_node_num = []
        for child in children:
            node_num = len(node_list)
            create_tree(root, child, node_list, sub_id_list,
                        leave_list, tokenizer, parent=root)
            root_children_node_num.append(len(node_list) - node_num)
        return root_children_node_num
    else:
        new_token = sub_token_list[0] if len(
            sub_token_list) <= 1 else tokenize_token
        new_node = AnyNode(
            id=id, token=new_token, data=node, parent=parent)

        if len(sub_token_list) > 1:
            sub_id_list.append(id)
            for sub_token in sub_token_list:
                id += 1
                AnyNode(id=id, token=sub_token, data=node, parent=new_node)
                node_list.append(sub_token)
                sub_id_list.append(id)

        for child in children:
            create_tree(root, child, node_list, sub_id_list,
                        leave_list, tokenizer, parent=new_node)


# traverse the AST tree to get all the nodes and edges
def get_node_and_edge(node, node_index_list, tokenizer, src, tgt, edge_attrs, variable_token_list, variable_id_list, token_dicts, token_list, token_ids, parent_next=None):
    token = node.token
    node_id = tokenizer.convert_tokens_to_ids(token)
    assert isinstance(node_id, int)
    node_index_list.append(node_id)
    # node_index_list.append([vocab_dict.word2id.get(token, UNK)])
    # find out all variables
    token_dicts[node.id] = node.token
    if not node.children and token not in javalang_special_tokens:
        token_list.append(token)
        token_ids.append(node.id)

    if token in ['VariableDeclarator', 'MemberReference']:
        if node.children:  # some chidren are comprised by non-utf8 and will be removed
            child_token = node.children[0].token
            if child_token == tokenize_token:
                # print(node.children[0])
                child_token += ' - '+node.children[0].data
            variable_token_list.append(child_token)
            variable_id_list.append(node.children[0].id)

    children = node.children
    is_split_node = (node.token == tokenize_token)

    for idx, child in enumerate(children[:-1]):
        # edge_attr = NextSib if not is_split_node else SplitNextSib
        if not is_split_node:
            edge_attr = NextSib
            src.append(child.id)
            tgt.append(children[idx+1].id)
            edge_attrs.append(edge_attr)

            tgt.append(child.id)
            src.append(children[idx+1].id)
            edge_attrs.append(edge_attr)

        if node.token == 'SwitchStatement' and child.token == 'SwitchStatementCase' and parent_next:
            if 'BreakStatement' in [i.token for i in child.children]:
                src.append(child.id)
                tgt.append(parent_next.id)
                edge_attrs.append(ControlOut)

                tgt.append(child.id)
                src.append(parent_next.id)
                edge_attrs.append(ControlOut)

    # Control Flow
    if node.token == 'ForStatement' or node.token == 'WhileStatement':
        # assert len(children) == 2 or (len(children)==3 and children[-1] == 'outer')
        if len(children) >= 2:
            src.append(children[1].id)
            tgt.append(children[0].id)
            edge_attrs.append(LoopNext)

            tgt.append(children[1].id)
            src.append(children[0].id)
            edge_attrs.append(LoopNext)

    if node.token == 'IfStatement':
        assert (len(children) == 2 or len(children) == 3) or len(children) == 0
        if len(children) == 3:
            # assert children[0].token == 'BinaryOperation'
            src.append(children[0].id)
            tgt.append(children[-1].id)
            edge_attrs.append(ConditionNext)

            tgt.append(children[0].id)
            src.append(children[-1].id)
            edge_attrs.append(ConditionNext)

    for idx, child in enumerate(children):
        # parent_type = Parent if not is_split_node else SplitParent
        # child_type = Child if not is_split_node else SplitChild
        src.append(node.id)
        tgt.append(child.id)
        edge_attrs.append(AST_EDGE)
        src.append(child.id)
        tgt.append(node.id)
        edge_attrs.append(AST_EDGE)
        parent_next = children[idx+1] if (idx+1 < len(children)) else None
        get_node_and_edge(child, node_index_list, tokenizer,
                          src, tgt, edge_attrs, variable_token_list, variable_id_list, token_dicts, token_list, token_ids, parent_next)


# generate pytorch_geometric input format data from ast
def get_pyg_data_from_ast(ast, tokenizer=ast_tokenizer):
    node_list = []
    sub_id_list = []  # record the ids of node that can be divide into multple subtokens
    leave_list = []  # record the ids of leave
    new_tree = AnyNode(id=0, token=None, data=None)
    root_children_node_num = create_tree(
        new_tree, ast, node_list, sub_id_list, leave_list, tokenizer)
    # print('root_children_node_num', root_children_node_num)
    x = []
    edge_src = []
    edge_tgt = []
    edge_attrs = []
    # record variable tokens and ids to add data flow edge in AST graph
    variable_token_list = []
    variable_id_list = []
    token_dicts = {}

    token_list, token_ids = [], []

    get_node_and_edge(new_tree, x, tokenizer, edge_src, edge_tgt, edge_attrs,
                      variable_token_list, variable_id_list, token_dicts, token_list, token_ids)

    visiulize_tree(new_tree)
    # add data flow edge
    variable_dict = {}
    for i in range(len(variable_token_list)):
        if variable_token_list[i] not in variable_dict:
            variable_dict.setdefault(
                variable_token_list[i], variable_id_list[i])
        else:
            edge_src.append(variable_dict.get(variable_token_list[i]))
            edge_tgt.append(variable_id_list[i])
            edge_attrs.append(NextUse)

            edge_tgt.append(variable_dict.get(variable_token_list[i]))
            edge_src.append(variable_id_list[i])
            edge_attrs.append(NextUse)
            variable_dict[variable_token_list[i]] = variable_id_list[i]

    for idx, item in enumerate(leave_list[:-1]):
        edge_src.append(item)
        edge_tgt.append(leave_list[idx+1])
        edge_attrs.append(NextToken)

        edge_tgt.append(item)
        edge_src.append(leave_list[idx+1])
        edge_attrs.append(NextToken)

    edge_index = [edge_src, edge_tgt]

    # TODO 第一个词不需要空格符号
    token_list[0] = token_list[0].lstrip('Ġ')
    token_idx = tokenizer.convert_tokens_to_ids(token_list[0])
    x[token_ids[0]] = token_idx

    return x, edge_index, edge_attrs, root_children_node_num, token_list, token_ids


def get_graph_from_source(code, tokenizer=ast_tokenizer):
    ast = parse_program(code)
    return get_pyg_data_from_ast(ast, tokenizer)


def get_subgraph_node_num(root_children_node_num, divide_node_num, max_subgraph_num):
    subgraph_node_num = []
    node_sum = 0
    real_graph_num = 0
    for num in root_children_node_num:
        node_sum += num
        if node_sum >= divide_node_num:
            subgraph_node_num.append(node_sum)
            node_sum = 0

    subgraph_node_num.append(node_sum)
    real_graph_num = len(subgraph_node_num)

    if real_graph_num >= max_subgraph_num:
        return subgraph_node_num[: max_subgraph_num], max_subgraph_num

    # print(len(subgraph_node_num))
    # if the last subgraph node num < divide_node_num/2, then put the last subgraph to the second to last subgraph
    if subgraph_node_num[-1] < divide_node_num/2:
        subgraph_node_num[-2] = subgraph_node_num[-2] + subgraph_node_num[-1]
        subgraph_node_num[-1] = 0
        real_graph_num -= 1

    # zero padding for tensor transforming
    for _ in range(real_graph_num, max_subgraph_num):
        subgraph_node_num.append(0)

    return subgraph_node_num, real_graph_num
