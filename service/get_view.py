#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging

import time
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, write_dot, to_agraph
from stanfordcorenlp import StanfordCoreNLP

from dao.data_acquire import get_corpus, get_news_from_sql
from definitions import STANFORD_NLP_PATH
from service.train_saying_word import read_say_words

__author__ = "charlene"
__time__ = "2019-05-15"


import logging


# corpus = get_corpus()
#
# stanford_chinese_location = r'../data/models/stanford-corenlp-full-2018-10-05/'
#
# with StanfordCoreNLP(stanford_chinese_location, port=9999, lang='zh', quiet=False, logging_level=logging.DEBUG) as nlp:
#     try_a = '中共中央总书记、国家主席、中央军委主席习近平在人民大会堂亲切会见大会代表，向他们表示热烈的祝贺，勉励他们再接再厉，为推进我国残疾人事业发展再立新功.'
#     print(nlp.ner(try_a))
#     print(nlp.dependency_parse(try_a))
#     # for article in corpus:
#     #     print(nlp.ner(article))
#     #     print(nlp.dependency_parse(article))
def get_stanford_nlp():

    #
    nlp = StanfordCoreNLP(STANFORD_NLP_PATH, memory='8g', port=9999, lang='zh', quiet=False, logging_level=logging.DEBUG)

    return nlp

def reconstruct_tree(tree_list, wordlist=None, with_name=False):
    """
    重构一棵树的表示
    :param tree_list: [(dep(依赖), gov(前驱), depd(后继)), (), ...]
    :param wordlist:
    :param with_name: 是否同时返回节点名（节点对应的词）
    :return: dict[key=父节点:value=[(子节点1，关系), (子节点2，关系)], ...])，~~每个元素索引等于对应节点的编号~~
    """

    # new_tree = [[] for i in range(len(tree_list) + 1)]
    new_tree = defaultdict(list)
    for dep, gov, depd in tree_list:

        if with_name:
            depd = (wordlist[depd - 1] if depd > 0 else 'root') + ':' + str(depd)
            gov = (wordlist[gov - 1] if gov > 0 else 'root') + ':' + str(gov)

        new_tree[gov].append((depd, dep))

    return new_tree

def reconstruct_tree_dictnode(tree_list):
    """
    重构一棵树的表示
    :param tree_list: [{'dep'(依赖):, 'governor'(前驱id):, 'governorGloss'(前驱word):,'dependent'(后继id):,
    'dependentGloss'(后继word):}, {}, ...]
    :return:  dict[key=父节点:value=[(子节点1_id，子节点1_word, 关系), (子节点2_id，子节点2_word, 关系)], ...])，~~每个元素索引等于对应节点的编号~~
    """

    # new_tree = [[] for i in range(len(tree_list) + 1)]
    new_tree = defaultdict(list)
    for node in tree_list:

        new_tree[node['governor']].append(node)

    return new_tree

def subtree_to_sentence(tree, reconstructed_tree, sub_root, wordlist=None):


    if wordlist:
        nodes = subtree(tree, reconstructed_tree, sub_root)

        words = [wordlist[idx[0]-1] for idx in sorted(nodes, lambda t: t[0])]

    else:
        nodes = subtree(tree, reconstructed_tree, sub_root, lambda t: t['dependent'])

        words = [n['dependentGloss'] for n in sorted(nodes, key=lambda i: i['dependent'])]

    return words


def plot_dep_tree(tree):
    g = nx.DiGraph()
    labels = defaultdict()

    for node,leafs in tree.items():
        for l in leafs:
            g.add_edge(node, l[0])
            labels[(node, l[0])] = l[1]

    # write_dot(g, 'test.dot')

    # pos = nx.spring_layout(g)
    pos = graphviz_layout(g, prog='dot')


    plt.figure()
    nx.draw(g, pos, edge_color='black', width=1, linewidths=1, \
            node_size=20, node_color='pink', alpha=0.7, font_size=8, \
            labels={node: node for node in g.nodes()})
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_color='red', font_size=8, alpha=0.3)
    plt.axis('off')
    plt.show()


def plot_dep_tree_dictnode(tree):
    g = nx.DiGraph()
    labels = defaultdict()

    for link in tree:
        g.add_edge(str(link['governor']) + ':' + link['governorGloss'], str(link['dependent']) + ':' + link['dependentGloss'])
        labels[(str(link['governor']) + ':' + link['governorGloss'], str(link['dependent']) + ':' + link['dependentGloss'])] \
            = link['dep']

    # write_dot(g, 'test.dot')

    # pos = nx.spring_layout(g)
    pos = graphviz_layout(g, prog='dot')


    plt.figure()
    nx.draw(g, pos, edge_color='black', width=1, linewidths=1, \
            node_size=20, node_color='pink', alpha=0.7, font_size=8, \
            labels={node: node for node in g.nodes()})
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels, font_color='red', font_size=8, alpha=0.3)
    plt.axis('off')
    plt.show()


def has_obj(nodes):
    """
    判断node list 中是否有宾语
    :return:
    """
    has = False
    for n in nodes:
        if 'obj' in n['dep']:
            has = True
            break
    return has


def subtree(tree, reconstructed_tree, sub_root_id, index_func=lambda t: t[0]):
    """

    :param tree:
    :param reconstructed_tree:
    :param sub_root_id:
    :param index_func:
    :return:
    """
    nodes = []
    unseen = [sub_root_id]

    tree_index = [i['dependent'] for i in tree] if type(tree[0]) == dict else [i for i in range(len(tree))]

    while len(unseen) > 0:
        current_n = unseen.pop()
        unseen += [index_func(t) for t in reconstructed_tree[current_n]]

        nodes.append(tree[tree_index.index(current_n)])

    return nodes


def views_observe(text, nlp, say_set):
    """
    输出text中，包含ner为person、其父节点或父节点的conj是"说"的同义词的句子和deptree，以观察view有哪些结构
    :param text:
    :param nlp:
    :param say_set:
    :return:
    """

    ss = []
    trees = []

    _props = {
        'annotators': 'tokenize, ssplit, pos, ner, depparse',
        'pipelineLanguage': 'zh',
        'outputFormat': 'json'
    }

    result_json = nlp.annotate(text, properties=_props)

    result_obj = json.loads(result_json)
    for sentence in result_obj['sentences']:
        say_words = dict([(i['index'], i['word']) for i in sentence['tokens'] if
                          i['word'] in say_set and i['pos'][0] == 'V'])  # V开头的动词

        if len(say_words) == 0:
            return [], []
        entity_name = set(['PERSON', 'ORGANIZATION', 'COUNTRY', 'DEMONYM', 'MISC', 'GPE'])
        persons = dict(
            [(i['index'], i['word']) for i in sentence['tokens'] if i['ner'] in entity_name]
        )

        if len(persons) == 0:
            return [], []

        dep = sentence['enhancedPlusPlusDependencies']
        dep_index = [i['dependent'] for i in dep]
        deptree = reconstruct_tree_dictnode(dep)

        seen = set()
        for ps in persons.keys():
            parent = dep[dep_index.index(ps)]
            p_conj = [i for i in dep if i['governor'] == parent['dependent'] and i['dependent'] in persons and i['dep'] == 'conj'] + \
                [parent]

            for p in p_conj:
                if p['governorGloss'] in say_set and p['governor'] not in seen:
                    seen.add(p['governor'])
                    ss.append(p['dependentGloss'] + ': ' + ''.join(subtree_to_sentence(dep, deptree, p['governor'])))
                    trees.append(subtree(dep, deptree, p['governor'], lambda t: t['dependent']))

    return ss, trees


def views_observe_2(text, nlp, say_set):
    """
    返回text中"说"的近义词的子树
    :param text:
    :param nlp:
    :param say_set:
    :return:
    """

    ss = []
    tree_sentence = []
    trees = []

    _props = {
        'annotators': 'tokenize, ssplit, pos, ner, depparse',
        'pipelineLanguage': 'zh',
        'outputFormat': 'json'
    }

    result_json = nlp.annotate(text, properties=_props)

    result_obj = json.loads(result_json)
    for sentence in result_obj['sentences']:
        say_words = dict([(i['index'], i['word']) for i in sentence['tokens'] if
                          i['word'] in say_set and i['pos'][0] == 'V'])  # V开头的动词

        if len(say_words) == 0:
            return [], [], []

        dep = sentence['enhancedPlusPlusDependencies']
        deptree = reconstruct_tree_dictnode(dep)


        for s in say_words:
            ss.append(say_words[s])
            tree_sentence.append(''.join(subtree_to_sentence(dep, deptree, s)))
            trees.append(subtree(dep, deptree, s, lambda t: t['dependent']))
    return ss, tree_sentence, trees



def get_particular_news():
    news = [
        """
        而上述不同供应商的配置方案，京东现场人员表示，会根据运营情况调整，因地制宜。
        """,
        """
        杰克逊也似乎在推特上向他的球迷告别，同时还表态称，虽然自己更愿意留在欧洲联赛或者去NBA发展，但他觉得这份合同（来CBA打球）仅仅是一份6个月的短合同，之后他还会有其他选择。
        """,
        """
        据欧洲篮球专家王健微博透露，这名法国后卫有超强的个人得分能力，堪称刷分机器。其个人能力与美籍后卫相比，也丝毫不逊色。
        """,
        """
        关于这份内部文件的真伪，日本文部科学省在第一时间宣称相关文件并不存在，但在已经辞职的元政务官和多位在职职员的证言下，本月15日又改口承认，确实有这样一份文件，并且承认在批准新设兽医学科过程中，受到了来自首相官邸的压力。
        """,
        """
        6月21日，MSCI在官网发布公告称，从明年6月起将中国A股纳入MSCI新兴市场指数和MSCI ACWI全球指数，这恐怕是近半年来中国资本市场上最令人振奋的消息。
A股早在2013年6月就已纳入新兴市场指数的候选列表中，但此后几年，都因为配额分配、资本流动限制、资本利得税等所谓原因而遭否决，尤其是在2016年第三次闯关失败后，中国投资者和相关监管部门似乎对“A股入摩”已心灰意冷，甚至连证监会分管国际合作的副主席方星海都在今年一月份的时候表示，“中国与MSCI在股指期货上的观点存在分歧，中国并不急于加入MSCI全球指数”。
然而事情最终出现了转机，今年3月，MSCI提出纳入A股的新方案——将A股的权重由原计划的1%降低至0.5%，并将指数纳入A股的数量由原来计划的448只减至169只，这一举动其实已经预示了A股今年大概率“入摩”。从6月21日宣布的结果来看，相比3月份的调整可以说还有惊喜，最终确定的权重为0.73%，股票数量为222只。
就具体的时间表而言，MSCI新兴市场A股纳入计划分两步走，第一步是在2018年5月按2.5%的指数纳入因子（index inclusion factor）给予A股0.37%的权重，第二步是在2018年8月按5%的因子将权重提高至计划的0.73%。从现在到A股正式进入MSCI新兴市场指数尚有一年时间，因此短期来看，这一事件不会马上起到提振国内股市的作用。另一方面，大部分机构预计本次“入摩”将为中国带来约1000亿美元的资金流入。相比于标的公司近2万亿美元的市值来说，这些资金并不能在市场上掀起太大的涟漪，只有当纳入因子进一步提高时（根据MSCI在2016年提出的A股纳入计划，纳入因子达到100%时，A股的权重将达到18.1%），“入摩”才可能在资金面上直接对A股市场有重大利好。
除了股价上的利好，“入摩”更重大的意义在于其给国内资本市场改革带来机遇与动力。MSCI在做出纳入决定前，需要广泛咨询国际机构投资者，这些投资者能够在全球范围内进行资产配置，他们最擅于比较各个国家的投资环境、资本市场对外资的友好程度。此前A股屡次碰壁就是因为国内金融市场上的QFII配额限制、QFII每月资本赎回限制、大面积股票停牌以及交易所需对A股相关金融产品预审等过多的管制与不确定性，给海外机构投资中国市场设下实质障碍，也让海外机构担心投入中国的资金的安全性。这种情况下，海外机构毫无疑问会选择“用脚投票”，避免参与中国市场。现在随着沪港通、深港通的开通，交易所产品预审的放松等，A股市场终于得到了国际机构的初步认可。想要“乘胜追击”，进一步吸引境外投资者，资本市场无疑还需要进一步的改革，否则可能错失好不容易打下的良好局面。
        """,
        """
        今日早评指出“创业板目前还没有回踩到位，所以今日还需要继续调整一下。今日主要是一个探底企稳的走势”。
        """,
        """
        从昨日市场走势看，又一次证明了彬哥之前所让大家关注的3180点主板可能回调的观点是正确的，而主板也确实在3180点附近出现了回调，对于这样的回调，彬哥认为是一个健康的事情，也是一个必须要出现的事情，现在唯一不能确定的就是在主板回调的时候中小创会不会出现企稳，会不会出现让投资者操作的机会，因此大家今天上午需要至少需要半天的时间去观察市场可能出现的结果，目前行情有一点可以确定的就是主板将会在3180点附近出现调整，因此大家万不可在这个时候再去追涨主板的个股（尤其是以保险银行为首的上证50个股），就周四午后市场的走势看，主板的调整势在必行，因此大家要做好主板调整的心理准备。当然还要做好市场全线调整的心理预期。
        """,
        """
    至于电池缩水，可能与刘作虎所说，一加手机5要做市面最轻薄大屏旗舰的设定有关。
        """,
        """
        此前的一加3T搭载的是3400mAh电池，DashCharge快充规格为5V/4A。
至于电池缩水，可能与刘作虎所说，一加手机5要做市面最轻薄大屏旗舰的设定有关。
按照目前掌握的资料，一加手机5拥有5.5寸1080P三星AMOLED显示屏、6G/8GB RAM，64GB/128GB ROM，双1600万摄像头，备货量“惊喜”。
根据京东泄露的信息，一加5起售价是xx99元，应该是在2799/2899/2999中的某个。

        """,
        """
        在A股纳入MSCI新兴市场指数，以及中国宏观环境持续稳定的双重因素鼓舞下，基金经理马磊表示，预计2017年下半年中国股市投资者情绪将有望转好。中国股市未来十年增长将主要由创新驱动。
       """,
        """
        对于乐视卖地的消息，绯闻的另一方万科表示不予回复。
        """,
        """
        “梅承认做得不够好”，BBC17日称，为平息怒火，梅当天抽出2小时，在唐宁街会见灾民和志愿者，并主持了一场政府应对火灾的会议。她承诺将亲自监督进行相关公共调查，拿出500万英镑支持灾民，并表示无家可归者将在3周内得到重新安置。
        """,
        """
                    英国与欧盟的“脱欧”谈判于19日正式开始。然而此时，英国首相特雷莎·梅正面临着空前的政治压力。不久前的大选失利让梅饱受诟病，而尘埃未落的伦敦城西“格伦费尔塔”大火激起的民怨，又给梅“火上浇油”。18日，多家英媒爆出，由于对梅失去信任，保守党党内正在酝酿一场“政变”。
        英国《每日电讯报》称，议会选举败北后，梅对14日“格伦费尔塔”火灾的无情和迟钝反应令她陷入巨大的政治危险。根据伦敦警方17日公布的数字，至少有58人被推定在火灾中丧生，随着搜寻工作继续进行，这一数字可能还会上升。路透社称，如果数字最终确定，“格伦费尔塔”火灾将成为二战后英国发生的最严重火灾。
        英国舆论沉浸在悲伤气氛中的同时，把矛头指向了首相梅的“冷漠”和应对不当。路透社17日称，火灾发生后，英国女王伊丽莎白二世和她的孙子威廉王子16日赴火灾发生地探望灾民和志愿者，女王17日又在自己91岁官方生日庆典上主持了1分钟的默哀仪式，并针对英国近来发生的数起事故“罕见”地呼吁民众“在哀伤中团结起来”。批评者指出，梅在灾后的表现和女王形成“鲜明对比”，显示出梅未能感受到公众情绪，且行动不坚决。
        英国“天空新闻网”17日称，梅在事发后视察火灾现场但未慰问灾民备受指责。作为补救，梅16日来到火灾发生地附近的一座教堂与当地居民见面。但抗议者在教堂外大喊“懦夫”“你不受欢迎”等口号，梅只好在警卫护送下匆匆离去。报道称，除英国女王外，工党领袖科尔宾也在第一时间去了现场并探望幸存者，他们的做法与梅形成“鲜明对比”。
        17日的英媒报道中充满了对梅的讽刺， 英国《每日镜报》头版以“两位领袖的故事”为题对比了梅和女王的灾后表现，并附上两幅截然不同的照片。一幅显示梅慰问火灾生还者受到警卫严密保护，另一幅则是女王与受灾社区居民亲切交谈的场景。
        “梅承认做得不够好”，BBC17日称，为平息怒火，梅当天抽出2小时，在唐宁街会见灾民和志愿者，并主持了一场政府应对火灾的会议。她承诺将亲自监督进行相关公共调查，拿出500万英镑支持灾民，并表示无家可归者将在3周内得到重新安置。
        然而，就在17日下午，首相官邸所在的唐宁街爆发了大规模抗议活动。英国《独立报》称，17日下午，大约1000名抗议者出现在唐宁街，高呼“科尔宾上台”“对抗保守党政府”。这场集会原本是抗议梅率领的保守党与爱尔兰民主统一党谈判联合组阁的，但后来加入了许多对梅应对火灾不力不满的民众。
        路透社称，梅决定提前大选，又未能让保守党在大选中获得绝对多数，已经让英国陷入自一年前“脱欧”公投以来最深刻的政治危机中。专栏作家、前保守党议员帕里斯认为，现在，梅应对火灾的行动表明，她缺乏判断力，“若无法重建公众信任，这个首相当不久”。
        《星期日泰晤士报》18日称，在梅领导的保守党内，人们对梅的信心不断下降，现在一些人甚至已经向她发出最后通牒，要求她在10天内证明她拥有自己所说的“领导能力”，否则就会采取行动赶她下台。报道透露，至少12名保守党议员已打算致函代表保守党后座议员的组织——1922委员会，建议对梅提出不信任动议。《星期日电讯报》18日也引述一些保守党“脱欧”派资深人士的话说，如果梅在即将展开的“脱欧”谈判中，背离原来的“硬脱欧”计划，他们就会立即对梅的领导权提出挑战。“脱欧”派议员警告说，任何让英国留在欧盟内的企图，或任何“偏航”的做法，都将在“一夜之间”触发“政变”。【记者 黄培昭 伊文】
        来源：新华网
        免责声明：本文仅代表作者个人观点，与环球网无关。其原创性以及文中陈述文字和内容未经本站证实，对本文以及其中全部或者部分内容、文字的真实性、完整性、及时性本站不作任何保证或承诺，请读者仅作参考，并请自行核实相关内容。
                """,
        """
        　　“之前的乐视一直分为3个体系：上市体系、非上市体系和汽车体系，5月底乐视由3个体系变成两个体系——上市体系和汽车体系。从那一天开始，乐视已经分裂，形成一个姓孙的乐视和姓贾的乐视。”资深家电分析师刘步尘对《中国经济周刊》分析。
        """,
        """
            　　文章导读： 供应商围堵追债、20多位高管离职、上千人被裁员、孤注一掷史上最大规模的降价……乐视的生态帝国风雨飘摇。
　　《中国经济周刊》 记者 侯隽| 北京报道
　　责编：周琦
　　（本文刊发于《中国经济周刊》2017年第24期）
　　供应商围堵追债、20多位高管离职、上千人被裁员、孤注一掷史上最大规模的降价……乐视的生态帝国风雨飘摇。
　　在乐视欲出售地产项目世茂·工三为自己“续命”之余，其创始人贾跃亭家族的持股规模也在持续下降。Wind资讯统计数据显示，2014年至今，贾跃亭家族在乐视网多次减持，涉及资金超过百亿元。尤其是2015年6月、2015年10月和今年1月的3次大规模减持后，曾持股近45%的贾跃亭，持股比例仅剩26.45%。
　　从“All In”蒙眼狂奔，到被讨债、围攻，再到被“拯救”、辞去总经理，经历了巨大风波的贾跃亭，其在乐视内部的影响力和权威无疑都已受到孙宏斌的挑战。如今，他能掌控的力量已经发生了巨大的变化。
　　“姓孙的乐视和姓贾的乐视”分裂为两个乐视
　　贾跃亭一直被外界视为乐视的灵魂人物，是他用4年时间，把一个视频网站，打造成为横跨电视、手机、汽车、金融、体育、影视等诸多板块的“生态王国”。
　　但是，这一切在2017年5月21日下午风云突变。
　　当日下午，乐视网发布第三届董事会第四十次会议决议公告称，为集中精力履行董事长职责，将工作重心集中于公司治理、战略规划及核心产品创新，提高公司决策效率，贾跃亭辞去公司总经理职务，专任公司董事长一职。
　　同时，经乐视网董事会提名委员会提名，聘请梁军担任乐视网总经理，向董事长及董事会汇报公司经营状况。乐视网成立以来，第一次有了专职总经理。
　　“之前的乐视一直分为3个体系：上市体系、非上市体系和汽车体系，5月底乐视由3个体系变成两个体系——上市体系和汽车体系。从那一天开始，乐视已经分裂，形成一个姓孙的乐视和姓贾的乐视。”资深家电分析师刘步尘对《中国经济周刊》分析。
　　5个月前，乐视由于一路“蒙眼狂奔”，导致整个生态体系资金链面临崩盘危机，被视为“白武士”的孙宏斌进场，后者带来的150亿元资金为乐视解了燃眉之急。
　　但是，天下没有免费的午餐，带着救命钱来的孙宏斌也是一个强势和高效的股东。
　　孙宏斌对乐视做的第一件事，就是推动乐视改革公司治理结构，把上市体系和非上市体系进行隔离。孙宏斌对此解释：“一定要让上市体系有完整的封闭性，资金是封闭的。”
　　对于“乐视姓贾还是姓孙”的问题，贾跃亭和孙宏斌这两个同样出身山西的商人都给出了答案。贾跃亭称这个问题压根儿不用回答，“如果你们相信那个谣言，孙总就不会投乐视了。”孙宏斌则说，“我要控制权干吗，累不累？乐视是贾跃亭的半条命，也是我的半条命，他要管，我当然也要管，但是我只是管治理结构、管理体系，其他的事情管不了。”同时，孙宏斌公开表示：“乐视汽车贾跃亭要怎么玩就怎么玩，他的主要精力也将放在汽车上。”
　　""",
    """
    “从这可以看出孙宏斌‘自觉’担任起了乐视发言人的角色，他在乐视的影响越来越大，成为制衡贾跃亭的人。”刘步尘说。
　　显然，孙宏斌带来了大笔资金，但是他也砌了一堵墙，即重点扶持能挣钱的项目，接近盈利的乐视电视变成重点，暂时无法盈利的体育、易到、手机业务，则成为可舍弃的项目。
　　乐视网作为中国A股最早上市的视频公司，一路发展成为创业板权重股。陷入危机后，乐视股价一路下跌，最低触及30元。
　　但是，贾跃亭也没有那么悲观。资料显示，乐视自成立以来，通过IPO、定向增发和发债，共融资91亿元。乐视不断壮大的同时，贾跃亭家族也在陆续减持股份。根据Wind资讯的统计，2012年至今，贾跃亭家族减持的乐视股票价值超过百亿元。
　　从公司公告披露的信息来看，贾跃亭本人一共有过3次大规模减持。
　　2015年6月1日至3日，贾跃亭减持约25亿元，减持后持股比例为42.30%。贾跃亭当时承诺，将全部套现金额无息借给上市公司，用于乐视日常经营。
　　2015年10月30日，贾跃亭协议转让1亿股乐视股票给鑫根基金，减持金额为32亿元，减持后持股比例为37.43%。
　　2017年1月16日，为“引入战略投资者、优化公司股权结构”，贾跃亭转让1.7亿股给天津嘉睿汇鑫企业管理有限公司，减持金额为60.41亿元，减持后的持股比例为26.45%。
　　不到两年的时间，贾跃亭持股比例由近45%降到26.45%，若加上其姐姐贾跃芳，贾跃亭家族减持涉及的金额超过百亿元。
　　除减持外，贾跃亭的股票还有一大部分用在了股权质押上。自2013年起，贾跃亭进行过至少38笔股权质押，累计获得的总金额超过311亿元。
　　那么，贾跃亭减持的钱都去了哪儿呢？
　　根据乐视公告给出的说法，这些钱的去处有二：将套现金额无息借给上市公司，用于乐视网日常经营；引入战略投资者、优化公司股权结构。
　　然而，据媒体报道，截至2015年12月31日，贾跃亭向乐视网提供的无息借款金额仅为20.71亿元。
　　人事大洗牌，千人被裁员或解约
　　显然，贾跃亭在一系列减持后，已经对乐视的主导权开始动摇。相应的，乐视开始人事大调整和大裁员。
　　融创宣布投资乐视150亿元后，做的第一件事就是推动乐视完善治理结构，派出公司负责风控和内审的刘淑青担任乐视董事，并在乐视网董事会中具有否决权。此外，融创还任命三位财务经理分赴乐视影业、乐视网和乐视致新。
　　从那时起，乐视经常爆出裁员或“人员优化”的消息。
　　最近的两次分别在5月18日和24日，乐视北美裁员约325人，仅剩60~80名员工；乐视非上市体系也开始裁员，乐视销服平台（手机业务）与乐视体育成为“重灾区”。
　　《中国经济周刊》根据公开资料统计，目前乐视北美、乐视非上市体系、加上乐视关联公司酷派，累计超过1000人被裁员或解约。据报道，乐视控股体系中，市场品牌中心与乐视体育的裁员幅度均达到70%；销售服务体系裁员幅度为50%；乐视网裁员幅度为10%；只有乐视影业、乐视致新暂未有裁员计划。最新的消息称，乐视移动员工目前有三种状态：离职、继续观望和协商赔偿。
　　不仅如此，6月10日多位乐视网员工还通过社交媒体透露，乐视网只把公积金、社保等五险一金缴纳到2017年3月份，4月及5月都处于欠缴状态，不少员工已经向有关部门投诉维权。此前，已有乐视体育、易到用车等乐视旗下公司员工公积金、社保断缴问题不断被曝光。
　　在孙宏斌所关注的乐视上市体系中，除了上市公司CEO、财务总监、非独立董事等核心职位进行过人事更迭外，高层和员工团队均比较稳定。
　　不过，贾跃亭主管的非上市体系似乎人心涣散。他花大力气“挖”来的人才纷纷离职，这些高管在乐视的任职时间大多在一年左右，最短的只有3个月。
　　最近有消息称，乐视员工已被猎头大规模拉黑，离职后就业艰难。一位已离职的乐视非上市体系员工向《中国经济周刊》记者表示，乐视一些员工现状确实如此，他自己目前就处于失业状态。
　　卖卖卖能否“续命”？
　　虽然不断辟谣裁员风波没有外界传得那么邪乎，但是乐视已经开始甩卖资产“续命”。
　　有消息称，乐视拟将旗下世茂·工三商业项目以40亿元出售给万科，目前双方正在商谈中。乐视CFO张巍在与乐视债权人沟通中透露了此消息，且对此事抱期待态度。张巍称，若项目成功出让，银行挤兑和供应商欠款会解决，部分银行可给乐视续贷，只要乐视业务稳步发展，乐视资金问题可以解决。
　　对于乐视卖地的消息，绯闻的另一方万科表示不予回复。
　　公开资料显示，乐视所持世茂·工三商业项目建筑面积为5万平方米，租赁面积约2.6万平方米，日客流量约2.5万人。2016年5月，乐视控股从上海世茂购得北京财富时代置业有限公司和北京百鼎新世纪商业管理有限公司100%股权，从而获得世茂·工三商业项目，彼时交易对价合计约29.72亿元。2016年11月，乐视将这两家公司股权质押给中信银行。
　　此外，乐视手机的出货目标也由最初的1300万部缩至现在的900万部左右，不到2016年近2000万部销量的一半。有消息称，乐视手机下调出货目标，原因是无法支付供应链的订单费用。在临近6·18之际，一家全国最大的手机通货平台将乐视手机下架。坊间预测，乐视手机业务或许很快会被卖掉。
　　“该卖的都卖掉”，孙宏斌早在公开场合提出过自己的观点。
        """,
        """
       此次“袭机”事件“惹怒”了俄罗斯。
       """,
        """
        俄罗斯参议院国防委员会副主席弗朗茨·克莱琴谢夫（Frants Klintsevich）称美军的行动是“挑衅行为”，实际上是对叙利亚的“军事侵略”。
        """,

        """
        	中新网6月19日电 据外媒报道，美国底特律一名男子1976年因为一根头发被定谋杀罪，监禁41年后，终于在协助下成功洗刷罪名，于本月15日获释。
据悉，61岁的沃金斯被控在1975年抢劫25岁女子伊薇特，并将其枪杀。当年20岁的沃金斯被警方逮捕，最后被以一级谋杀罪定罪，而他被定罪的关键仅是在现场的一根头发。
西密西根大学库利法学院的“清白项目”为了协助更多蒙受冤狱之苦的受刑人，积极帮助沃金斯，发现当年警局实验室的分析师基于在现场找到的一根头发，而认定该案与沃金斯有关。
清白专案总监马拉米契尔说，根据头发定罪不是建立在科学的基础上。该组织今年1月要求法院撤销沃金斯的定罪。
沃金斯等待41年后终于真相大白，被获撤销罪名后，走出底特律下城的韦恩郡监狱。他说：“这真梦幻，令人难以置信，但我感觉很好，我早料到有这一天，但想不到要等41年那么长。”
免责声明：本文仅代表作者个人观点，与环球网无关。其原创性以及文中陈述文字和内容未经本站证实，对本文以及其中全部或者部分内容、文字的真实性、完整性、及时性本站不作任何保证或承诺，请读者仅作参考，并请自行核实相关内容。
        """

    ]

    return news

def get_entity_mod(tree, entity):
    """
    返回一个有修饰词的完整实体（人名、组织等）
    :param tree:
    :param entity:
    :return:
    """
    return entity['dependentGloss']

def get_views(text, nlp, say_set):
    """
    获取sentence中的观点
    :param sentence:
    :param nlp:
    :param say_set:
    :return: [(person1, say_word1, view1),(person2, say_word2, view2),...]
    """
    views = []

    _props = {
        'annotators': 'tokenize, ssplit, pos, ner, depparse',
        'pipelineLanguage':'zh',
        'outputFormat': 'json'
    }

    result_json = nlp.annotate(text, properties=_props)


    result_obj = json.loads(result_json)
    for sentence in result_obj['sentences']:
        say_words = dict([(i['index'], i['word']) for i in sentence['tokens'] if i['word'] in say_set and i['pos'][0] == 'V'])  # V开头的动词

        if len(say_words) == 0:
            return views

        # entity_name = set(['ORGANIZATION', 'PERSON', 'COUNTRY', 'GPE', 'MISC'])
        # persons = dict(
        #     [(i['index'], i['word']) for i in sentence['tokens'] if i['ner'] in entity_name]
        # )

        # if len(persons) == 0:
        #     return views
        tokens = sentence['tokens']

        dep = sentence['enhancedPlusPlusDependencies']
        deptree = reconstruct_tree_dictnode(dep)

        dep_index = [i['dependent'] for i in dep]

        tk_index = [i['index'] for i in tokens]



        say_words = [n for n in sentence['tokens'] if n['word'] in say_set]

        seen = set()

        for s in say_words:

            if s['index'] not in seen:
                s_node = dep[dep_index.index(s['index'])]

                subj = [n for n in deptree[s_node['dependent']] if
                        'subj' in n['dep']]  # or tokens[tk_index.index(n['dependent'])]['ner'] in entity_name]

                if len(subj) > 0:

                    if len(subj) > 1:
                        logging.ERROR('More than one subject in sentence {}, \nfound subjects: {}'.format(
                            ''.join([w['originalText'] for w in sentence['tokens']]),
                            subj
                        ))

                    subj = min(subj, key=lambda t: abs(t['dependent'] - s['index']))

                    # if tokens[tk_index.index(subj['dependent'])]['pos'] != 'PN':  # subject is not pronoun

                    for n in deptree[s['index']]:
                        if n['dep'] == 'conj':
                            s_n = [c for c in deptree[n['dependent']] if 'subj' in c['dep']]
                            adv_n = [c for c in deptree[n['dependent']] if c['dep'] == 'advmod']

                            if len(s_n) > 0:

                                if len(s_n) > 1:
                                    logging.ERROR(
                                        'More than one subject in sub sentence {}, \nfound subjects: {}'.format(
                                            ''.join(subtree_to_sentence(dep, deptree, s_node['dependent'])),
                                            s_n
                                        ))

                                s_n = s_n[0]

                                if tokens[tk_index.index(s_n['dependent'])]['pos'] == 'PN':
                                    view = ';'.join(
                                        [''.join(subtree_to_sentence(dep, deptree, child['dependent']))
                                         for child in deptree[n['dependent']] if child['dep'] == 'ccomp']
                                    )
                                    if view:
                                        views.append((get_entity_mod(dep, subj), n['dependentGloss'], view))

                            elif len(adv_n) > 0:
                                view = ';'.join(
                                    [''.join(subtree_to_sentence(dep, deptree, child['dependent']))
                                     for child in deptree[n['dependent']] if child['dep'] == 'ccomp']
                                )
                                if view:
                                    views.append((get_entity_mod(dep, subj), n['dependentGloss'], view))



                        elif n['dep'] == 'ccomp':

                            view = ''.join(subtree_to_sentence(dep, deptree, n['dependent']))

                            views.append((get_entity_mod(dep, subj), s['word'], view))


                seen.add(s['index'])

        return views

def _test_views_observe():
    # news_set = get_news_from_sql(num=500, need_token=False)

    news_set = get_particular_news()

    say_sets = read_say_words()

    nlp = get_stanford_nlp()

    for new in news_set:
        ss, trees = views_observe(new, nlp, say_sets)

        if ss:
            print('The news :\n\t{}\n contains views:\n\t{} \nviews\' trees are \n\t{}'.format(new, ss, trees))

        for tree in trees:
            plot_dep_tree_dictnode(tree)


def test_views_observe_2():
    news_set = get_news_from_sql(num=500, need_token=False)

    # news_set = get_particular_news()

    say_sets = read_say_words()

    nlp = get_stanford_nlp()

    for new in news_set:
        ss, sentence, trees = views_observe_2(new, nlp, say_sets)

        if ss:
            print('The news :\n\t{}\n contains views:\n\tsay word is: {} \nviews\' are \n\t{}'.format(new, ss, sentence))

        for tree in trees:
            plot_dep_tree_dictnode(tree)


def _test():
    samples = get_particular_news()

    say_sets = read_say_words()

    nlp = get_stanford_nlp()


    for news in samples:

        print('\n\nview in news:{} \n are:\n {}'.format(news, get_views(news, nlp, say_sets)))

#
# 问题：
# 1. nlp输入是sentence，语料库是一整篇文章，如何分句？要不要标点符号？
# 2. 从句格式？（1、"说"是谓语，在dependency中是root，2、"说"的内容，在从句中）
# 3. 说是谓语的情况下，是否有：1、多个主语 2、多个谓语中包含多个说的同义词 -- 已解决，貌似有其他问题，想不起来了
# 4. 说的同义词效果不好
