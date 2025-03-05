# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import difflib
import os
import sys
from collections import namedtuple
from dataclasses import fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
from openvino import Core, Model, Type, properties, save_model
from openvino_tokenizers import convert_tokenizer
from openvino_tokenizers.constants import ORIGINAL_TOKENIZER_CLASS_NAME, rt_info_to_hf_attribute_map
from openvino_tokenizers.utils import TokenzierConversionParams, get_hf_tokenizer_attribute
from tokenizers.models import Unigram
from transformers import AutoTokenizer

from tests.utils import get_hf_tokenizer


if os.environ.get("OV_TOKENIZERS_TESTS_PRINT_WHOLE_DIFF"):
    np.set_printoptions(threshold=sys.maxsize)

core = Core()

eng_test_strings = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
    "<s>[INST] <<SYS>> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. <</SYS>> You will act as a Christian, and fully summarize following text:\nSometimes it's nice to take a minute in the pew by yourself beforehand. You have this beautiful church probably almost all to yourself. Can you feel its energy resonating through you? Can you feel the majesty of the Lord's kingdom and how you're a part of it? Take a moment to kneel and pray with your head down and hands clasped together. Reflect on your faith and how you feel currently. Think about how you've been responding to God's call and how you've been living in the light of his love. When the priest is ready for you, of course. You'll probably see him there by his lonesome or someone else walk out just before you. Sit down either across from him or behind the screen -- it's totally up to you whether or not you prefer to remain anonymous. He won't treat you any differently either way. Make the sign of the cross upon his prompt, saying, \"Bless me, Father, for I have sinned. It has been 10 years since my last confession.\" This is your standard, traditional phrasing. However, if you just sit down and say hello, that's fine, too. The priest knows what he's doing. The Byzantine Rite is a bit different. The priest may sit to your side and put his epitrachelion on your head. He may then also do the Prayer of Absolution. But the idea remains the exact same -- just go wherever he takes you. Once you sit down and you've made the sign of the cross, just sit back and follow the priest's lead. He'll ask you how long it's been since your last confession (if you don't voluntarily offer that information), how you are feeling, maybe how your faith is going, and then ask you what sins you would like to talk about with him and God. It's just a casual conversation! Do not fret. There is absolutely zero pressure on your part. Again, as long as you come there with the intention of leaving with a clean heart, you're more than welcome in the church. There is no wrong way to go about confession! This part is intimidating, but think about it this way: the priest you're talking to has probably heard just about everything before. Whatever you have to say will not blow his mind. So when he asks, start rattling them off, from the most serious to the least. If he asks any questions, answer them, but do not feel the need to go into detail. A simple, \"I did so and so,\" will suffice. Your priest is going to be very understanding. If you don't remember the exact timeframe, that's fine. If you don't remember your motivation, that's fine. All your priest cares about is that you're being as honest as possible and that your heart is in the right place. He'll talk you through everything, possibly asking about your intentions, but mainly just letting you know that God loves you, sin and all. If he has any ideas to bring you closer to God, he may suggest them at this juncture. He's there to help, after all. He will then ask you to make an Act of Contrition. That goes like this: My God, I am sorry for my sins with all my heart.In choosing to do wrong and failing to do good,I have sinned against You whom I should loveabove all things. I firmly intend, with your help,to do penance, to sin no more, andto avoid whatever leads me to sin.Our Savior Jesus Christ suffered and died for us.In his name, my God, have mercy.If you are a Roman Catholic, your act of contrition will go like this: Oh my God, I am very sorry for having offended thee. But most of all, because they offend you, my God, who is all good and deserving of all my love. I firmly resolve with the help of thy grace, to sin no more, and to avoid the near occasion of sin. Amen. Don't worry! It won't be anything huge. Take the absolution to heart -- you now have a brand new, clean slate to work with. \"Penance\" is your expression of regret and repentance, showing God that you're truly sorry and that you wish for nothing more than to be forgiven. Thanks. [/INST]",
    # Qwen tests
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
]
multilingual_test_strings = [
    "Тестовая строка!",
    "Testzeichenfolge?",
    "Tester, la chaîne...",
    "測試字符串",
    "سلسلة الاختبار",
    "מחרוזת בדיקה",
    "Сынақ жолы á",
    "رشته تست",
    # Qwen test
    "介绍下清华大学",
    "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
    "糕点商店里原本有三种蛋糕：草莓奶油蛋糕，巧克力椰蓉蛋糕，和红丝绒布朗尼蛋糕。如名字所描述的那样，每种蛋糕都有两种成分：草莓奶油蛋糕包含草莓和奶油两个成分，巧克力椰蓉蛋糕包含巧克力和椰蓉两种成分，红丝绒布朗尼蛋糕包含红丝绒和布朗尼两种成分。在蛋糕制作完成后，往往每一种成分的材料都会有所剩余。为了减少浪费，商店常常会把多出来的成分两两搭配，做成新的小商品卖出去。比如草莓和巧克力可以做成草莓味巧克力酱，布朗尼和椰蓉可以做成布朗尼椰蓉饼干。以此类推可知，如果所有的成分都可以两两组合，那么最终商店能做出哪些小商品出来？",
    "桌子有左中右3个抽屉；张三，李四，王五，赵六都看到桌子上有一袋巧克力。张三让李四和王五出门后，在赵六面前把这袋巧克力放进了右抽屉；王五回来后，张三让赵六出门去找李四，并在王五面前从左抽屉拿出一盒饼干放进中抽屉里；等李四和赵六返回，张三又让王五和赵六出去买酱油，等二人走后，他告诉李四刚才已将一盒饼干放进中抽屉；张三等了很久，发现王五和赵六还没回来，就派李四去寻找，可最后只有王五和李四回来了。王五告诉张三，一开始他们没有找到卖酱油的店，所以只好分头去买，后来赵六走丢了；回来的路上，王五碰上了李四，两人便先赶了回来。于是，张三让两人出门去找赵六；为防再次走丢，张三叮嘱李四和王五要时刻同行，就算酱油买不到，也要找回赵六。结果，李四和王五在外面找到了赵六，发现他已经买了酱油。三人觉得张三从来不出门跑腿，十分气愤，讨论并达成共识，回去见到张三后，不要告诉他买到了酱油的事情，并让王五把酱油藏到自己的背包里。等三人一同回来后，他们按照计划谎称没有买到酱油，并希望张三以后买东西也要一同出门，不能偷懒，张三答应了。当大家最后站在桌子前，四人分别写下自己知道的物品清单和物品所在位置。问，这四人写下的物品和位置信息是否一致，为什么？",
    "折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。请赏析以上内容的精妙之处。",
]
emoji_test_strings = [
    "😀",
    "😁😁",
    "🤣🤣🤣😁😁😁😁",
    "🫠",  # melting face
    "🤷‍♂️",
    "🤦🏼‍♂️",
]
misc_strings = [
    "",
    b"\x06".decode(),  # control char
    " ",
    " " * 10,
    " " * 256,  # from llama3/stablecode vocab
    "\n",
    " \t\n",
]
chat_messages = [
    [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]
]

wordpiece_models = [
    "bert-base-multilingual-cased",
    "cointegrated/rubert-tiny2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/mobilebert-uncased",
    "rasa/LaBSE",
]
bpe_models = [
    "Xenova/gpt-4o",
    "NousResearch/Meta-Llama-3-8B-Instruct",
    # "meta-llama/Meta-Llama-3-8B",  # cannot be part of the CI
    "tiiuae/falcon-7b",
    "stabilityai/stablecode-completion-alpha-3b-4k",
    "koalajun/Gemma-2-9b-it-Ko-Crypto-Translate",
    "roberta-base",
    "facebook/opt-66b",
    "gpt2",
    "ai-forever/rugpt3large_based_on_gpt2",
    "facebook/galactica-120b",
    "microsoft/deberta-base",
    "bigscience/bloom",
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "Salesforce/codegen-16B-multi",
    "stabilityai/stablelm-2-1_6b",
    "deepseek-ai/deepseek-coder-6.7b-instruct",  # sentencepiece tokenizer without .model file fallback to fast BPE
    "allenai/OLMo-1B-hf",
    "answerdotai/ModernBERT-base",
    # "google/flan-t5-xxl",  # needs Precompiled/CharsMap
    # "jinmang2/textcnn-ko-dialect-classifier",  # Needs Metaspace Pretokenizer
    # "hyunwoongko/blenderbot-9B",  # hf script to get fast tokenizer doesn't work
]
sentencepiece_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "openbmb/MiniCPM-V-2",  # have additional dependencies: deepspeed, peft, peft
    "baichuan-inc/Baichuan2-7B-Chat",
    "camembert-base",
    "NousResearch/Llama-2-13b-hf",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "xlnet-base-cased",
    # "THUDM/chatglm3-6b",  # _pad doesn't support padding side - broke in 4.45
    "t5-base",
    "facebook/musicgen-small",
    "rinna/bilingual-gpt-neox-4b",
    "microsoft/Phi-3-mini-128k-instruct",
    "mlx-community/quantized-gemma-7b-it",
]
tiktiken_models = [
    "Qwen/Qwen-14B-Chat",
    # "Salesforce/xgen-7b-8k-base",  # not compatible with transformers 4.44.0
    "THUDM/glm-4-9b-chat",
]
wordlevel_models = ["cisco-ai/mini-bart-g2p"]


def get_tokenizer(hf_tokenizer, add_special_tokens=True, use_max_padding=False, use_sentencepiece_backend=False):
    ov_tokenizer = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=False,
        add_special_tokens=add_special_tokens,
        use_max_padding=use_max_padding,
        use_sentencepiece_backend=use_sentencepiece_backend,
    )
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    return hf_tokenizer, compiled_tokenizer


def get_tokenizer_detokenizer(
    hf_tokenizer,
    streaming_detokenizer=False,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=None,
    use_sentencepiece_backend=False,
):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer,
        with_detokenizer=True,
        streaming_detokenizer=streaming_detokenizer,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        use_sentencepiece_backend=use_sentencepiece_backend,
    )
    compiled_tokenizer = core.compile_model(ov_tokenizer)
    compiled_detokenizer = core.compile_model(ov_detokenizer)
    return hf_tokenizer, compiled_tokenizer, compiled_detokenizer


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_left: "left_pad" if is_left else "right_pad")
def use_left_padding(request):
    return request.param


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_max: "max_pad" if is_max else "min_pad")
def use_max_padding(request):
    return request.param


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_wordpiece_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session", params=wordpiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_wordpiece_tokenizers_with_padding_sides(request, use_left_padding):
    return get_hf_tokenizer(request, left_padding=use_left_padding)


@pytest.fixture(scope="session", params=[True, False], ids=lambda is_fast: "Fast" if is_fast else "Slow")
def is_fast_tokenizer(request):
    return request.param


@pytest.fixture(scope="session", params=[True, False], ids=lambda to_add: "add_tokens" if to_add else "no_add_tokens")
def do_add_special_tokens(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda do_skip: "skip_tokens" if do_skip else "no_skip_tokens"
)
def do_skip_special_tokens(request):
    return request.param


@pytest.fixture(
    scope="session", params=[True, False], ids=lambda do_clean: "clean_spaces" if do_clean else "no_clean_spaces"
)
def do_clean_up_tokenization_spaces(request):
    return request.param


@pytest.fixture(scope="session", params=[True, False], ids=lambda do_clean: "sp_backend" if do_clean else "")
def is_sentencepiece_backend(request):
    return request.param


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_sentencepiece_tokenizers(request, is_fast_tokenizer, is_sentencepiece_backend):
    if not is_fast_tokenizer and not is_sentencepiece_backend:
        pytest.skip("Legacy tokenizer must use Sentencepiece backend.")

    hf_tokenizer = get_hf_tokenizer(request, fast_tokenizer=is_fast_tokenizer, trust_remote_code=True)
    if not hf_tokenizer.is_fast and is_fast_tokenizer:
        pytest.skip("Fast tokenizer should use Rust backend.")
    if (
        is_fast_tokenizer
        and not is_sentencepiece_backend
        and isinstance(hf_tokenizer.backend_tokenizer.model, Unigram)
    ):
        pytest.skip("Unigram model supports only sentencepiece backend.")

    return hf_tokenizer


@pytest.fixture(scope="session", params=sentencepiece_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_sentencepiece_tokenizers_with_padding_sides(
    request, use_left_padding, is_fast_tokenizer, is_sentencepiece_backend
):
    if not is_fast_tokenizer and not is_sentencepiece_backend:
        pytest.skip("Legacy tokenizer must use sentencepiece backend.")

    hf_tokenizer = get_hf_tokenizer(request, left_padding=use_left_padding, trust_remote_code=True)
    if not hf_tokenizer.is_fast and is_fast_tokenizer:
        pytest.skip("Fast tokenizer should use Rust backend.")
    if (
        is_fast_tokenizer
        and not is_sentencepiece_backend
        and isinstance(hf_tokenizer.backend_tokenizer.model, Unigram)
    ):
        pytest.skip("Unigram model supports only sentencepiece backend.")

    return hf_tokenizer


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_bpe_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session", params=bpe_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_bpe_tokenizers_with_padding_sides(request, use_left_padding):
    hf_tokenizer = get_hf_tokenizer(request, left_padding=use_left_padding)
    return hf_tokenizer


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_tiktoken_tokenizers(request):
    return get_hf_tokenizer(request, trust_remote_code=True)


@pytest.fixture(scope="session", params=tiktiken_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_tiktoken_tokenizers_with_padding_sides(request, use_left_padding):
    hf_tokenizer = get_hf_tokenizer(request, trust_remote_code=True, left_padding=use_left_padding)
    return hf_tokenizer


@pytest.fixture(scope="session", params=wordlevel_models, ids=lambda checkpoint: checkpoint.split("/")[-1])
def hf_wordlevel_tokenizers(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session")
def wordpiece_tokenizers(hf_wordpiece_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_wordpiece_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def wordpiece_tokenizers_with_padding_options(
    hf_wordpiece_tokenizers_with_padding_sides, do_add_special_tokens, use_max_padding
):
    if use_max_padding and getattr(hf_wordpiece_tokenizers_with_padding_sides, "model_max_length") > 2**31:
        pytest.skip("Cannot test max_padding=True for tokenizer without max length.")

    return get_tokenizer(
        hf_wordpiece_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
        use_max_padding=use_max_padding,
    )


@pytest.fixture(scope="session")
def wordpiece_tokenizers_detokenizers(
    hf_wordpiece_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    return get_tokenizer_detokenizer(
        hf_wordpiece_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(scope="session")
def bpe_tokenizers(hf_bpe_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_bpe_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def bpe_tokenizers_with_padding_options(hf_bpe_tokenizers_with_padding_sides, do_add_special_tokens, use_max_padding):
    if use_max_padding and getattr(hf_bpe_tokenizers_with_padding_sides, "model_max_length") > 2**31:
        pytest.skip("Cannot test max_padding=True for tokenizer without max length.")

    return get_tokenizer(
        hf_bpe_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
        use_max_padding=use_max_padding,
    )


@pytest.fixture(scope="session")
def bpe_tokenizers_detokenizers(hf_bpe_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces):
    return get_tokenizer_detokenizer(
        hf_bpe_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(scope="session")
def sentencepice_tokenizers(hf_sentencepiece_tokenizers, do_add_special_tokens, is_sentencepiece_backend):
    return get_tokenizer(
        hf_sentencepiece_tokenizers,
        add_special_tokens=do_add_special_tokens,
        use_sentencepiece_backend=is_sentencepiece_backend,
    )


@pytest.fixture(scope="session")
def sentencepiece_tokenizers_with_padding_options(
    hf_sentencepiece_tokenizers_with_padding_sides, do_add_special_tokens, use_left_padding, is_sentencepiece_backend
):
    if (
        hf_sentencepiece_tokenizers_with_padding_sides.name_or_path in ("THUDM/chatglm2-6b", "THUDM/chatglm3-6b")
        and not use_left_padding
    ):
        pytest.skip("chatglm supports left padding only")
    if hf_sentencepiece_tokenizers_with_padding_sides.name_or_path == "THUDM/chatglm2-6b" and do_add_special_tokens:
        pytest.skip("chatglm2 never adds special tokens")
    if (
        hf_sentencepiece_tokenizers_with_padding_sides.name_or_path == "THUDM/chatglm3-6b"
        and not do_add_special_tokens
    ):
        pytest.skip("chatglm3 always adds special tokens")

    return get_tokenizer(
        hf_sentencepiece_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
    )


@pytest.fixture(scope="session")
def sentencepice_tokenizers_detokenizers(
    hf_sentencepiece_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces, is_sentencepiece_backend
):
    # chatglm2 always skips special tokens, chatglam3 always not skip
    if hf_sentencepiece_tokenizers.name_or_path == "THUDM/chatglm2-6b" and not do_skip_special_tokens:
        pytest.skip("chatglm2 always skips special tokens")
    if hf_sentencepiece_tokenizers.name_or_path == "THUDM/chatglm3-6b" and do_skip_special_tokens:
        pytest.skip("chatglm3 always adds special tokens")

    return get_tokenizer_detokenizer(
        hf_sentencepiece_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
        use_sentencepiece_backend=is_sentencepiece_backend,
    )


@pytest.fixture(scope="session")
def tiktoken_tokenizers(hf_tiktoken_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_tiktoken_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def tiktoken_tokenizers_with_padding_options(
    hf_tiktoken_tokenizers_with_padding_sides, do_add_special_tokens, use_max_padding, use_left_padding
):
    if use_max_padding and getattr(hf_tiktoken_tokenizers_with_padding_sides, "model_max_length") > 2**31:
        pytest.skip("Cannot test max_padding=True for tokenizer without max length.")
    if not use_left_padding and hf_tiktoken_tokenizers_with_padding_sides.name_or_path == "THUDM/glm-4-9b":
        pytest.skip("chatglm supports left padding only")
    return get_tokenizer(
        hf_tiktoken_tokenizers_with_padding_sides,
        add_special_tokens=do_add_special_tokens,
        use_max_padding=use_max_padding,
    )


@pytest.fixture(scope="session")
def tiktoken_tokenizers_detokenizers(hf_tiktoken_tokenizers, do_skip_special_tokens):
    return get_tokenizer_detokenizer(
        hf_tiktoken_tokenizers, skip_special_tokens=do_skip_special_tokens, clean_up_tokenization_spaces=False
    )


@pytest.fixture(scope="session")
def wordlevel_tokenizers(hf_wordlevel_tokenizers, do_add_special_tokens):
    return get_tokenizer(hf_wordlevel_tokenizers, add_special_tokens=do_add_special_tokens)


@pytest.fixture(scope="session")
def wordlevel_tokenizers_detokenizers(
    hf_wordlevel_tokenizers, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    return get_tokenizer_detokenizer(
        hf_wordlevel_tokenizers,
        skip_special_tokens=do_skip_special_tokens,
        clean_up_tokenization_spaces=do_clean_up_tokenization_spaces,
    )


@pytest.fixture(
    scope="session", params=["openlm-research/open_llama_3b_v2"], ids=lambda checkpoint: checkpoint.split("/")[-1]
)
def hf_tokenizers_for_streaming(request):
    return get_hf_tokenizer(request)


@pytest.fixture(scope="session")
def sentencepiece_streaming_tokenizers(hf_tokenizers_for_streaming):
    return get_tokenizer_detokenizer(
        hf_tokenizers_for_streaming, streaming_detokenizer=True, use_sentencepiece_backend=True
    )


def print_diff(left, right) -> str:
    left = str(left.reshape(-1)).split("\n")
    right = str(right.reshape(-1)).split("\n")

    diff = "\n".join(difflib.ndiff(left, right))
    return f"\n{diff}"


def check_tokenizer_output(
    tokenizers: Tuple,
    test_string: Union[str, List[str]],
    skip_missing_outputs: bool = False,
    hf_tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    calculate_diff: bool = False,
) -> None:
    hf_tokenizer, ov_tokenizer = tokenizers
    hf_tokenizer_kwargs = {} if hf_tokenizer_kwargs is None else hf_tokenizer_kwargs

    if isinstance(test_string, str):
        test_string = [test_string]

    hf_tokenized = hf_tokenizer(test_string, return_tensors="np", truncation=True, **hf_tokenizer_kwargs)
    ov_tokenized = ov_tokenizer(test_string)

    for output_name, hf_result in hf_tokenized.items():
        if output_name not in ov_tokenized and skip_missing_outputs:
            continue

        assert output_name in ov_tokenized, f"OV Tokenizer missing output: {output_name}"
        ov_result = ov_tokenized[output_name]

        outputs = f"\n{hf_result}\n{ov_result}"
        diff = print_diff(hf_result, ov_result) if calculate_diff and ov_result.shape != hf_result.shape else outputs
        assert ov_result.shape == hf_result.shape, diff
        assert np.all(ov_result == hf_result), outputs


def check_detokenizer_output(
    detokenizers: Tuple,
    test_string: Union[str, List[str]],
    hf_detokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    hf_tokenizer, _, ov_detokenizer = detokenizers
    hf_detokenizer_kwargs = {} if hf_detokenizer_kwargs is None else hf_detokenizer_kwargs

    token_ids = hf_tokenizer(test_string, return_tensors="np", padding=True).input_ids
    hf_output = hf_tokenizer.batch_decode(token_ids, **hf_detokenizer_kwargs)
    ov_output = ov_detokenizer(token_ids.astype("int32"))["string_output"].tolist()

    assert ov_output == hf_output


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers(wordpiece_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        wordpiece_tokenizers,
        test_string=test_string,
        skip_missing_outputs=False,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_wordpiece_tokenizers_multiple_strings(
    wordpiece_tokenizers_with_padding_options, test_string, do_add_special_tokens, use_max_padding
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": "max_length" if use_max_padding else True,
    }
    check_tokenizer_output(
        wordpiece_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=False,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_wordpiece_model_detokenizer(
    wordpiece_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        wordpiece_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_tokenizer(sentencepice_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        sentencepice_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_chat",
    chat_messages,
)
def test_sentencepiece_model_tokenizer_chat(sentencepice_tokenizers, test_chat, do_add_special_tokens):
    hf_tokenizer, ov_tokenizer = sentencepice_tokenizers
    if hf_tokenizer.chat_template is None:
        pytest.skip("No chat template")

    from jinja2 import TemplateError

    try:
        test_string = hf_tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    except TemplateError:
        # filter system message
        test_string = hf_tokenizer.apply_chat_template(test_chat[1:], tokenize=False, add_generation_prompt=True)

    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        sentencepice_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_sentencepiece_tokenizers_multiple_strings(
    sentencepiece_tokenizers_with_padding_options, test_string, do_add_special_tokens
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": True,
    }
    check_tokenizer_output(
        sentencepiece_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_sentencepiece_model_detokenizer(
    sentencepice_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        sentencepice_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_hf_bpe_tokenizers_outputs(bpe_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        bpe_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )


@pytest.mark.parametrize(
    "test_chat",
    chat_messages,
)
def test_bpe_model_tokenizer_chat(bpe_tokenizers, test_chat, do_add_special_tokens):
    hf_tokenizer, ov_tokenizer = bpe_tokenizers
    if hf_tokenizer.chat_template is None:
        pytest.skip("No chat template")

    test_string = hf_tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        bpe_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_bpe_tokenizers_multiple_strings(
    bpe_tokenizers_with_padding_options, test_string, do_add_special_tokens, use_max_padding
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": "max_length" if use_max_padding else True,
    }
    check_tokenizer_output(
        bpe_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_bpe_detokenizer(
    bpe_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        bpe_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_tokenizers(tiktoken_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        tiktoken_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )


@pytest.mark.parametrize(
    "test_chat",
    chat_messages,
)
def test_tiktoken_model_tokenizer_chat(tiktoken_tokenizers, test_chat, do_add_special_tokens):
    hf_tokenizer, ov_tokenizer = tiktoken_tokenizers
    if hf_tokenizer.chat_template is None:
        pytest.skip("No chat template")

    test_string = hf_tokenizer.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens}
    check_tokenizer_output(
        tiktoken_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,  # chatglm has token_type_ids output that we omit
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        eng_test_strings,
        multilingual_test_strings,
        emoji_test_strings,
        misc_strings,
    ],
)
def test_hf_tiktoken_tokenizers_multiple_strings(
    tiktoken_tokenizers_with_padding_options, test_string, do_add_special_tokens
):
    hf_tokenizer_kwargs = {
        "add_special_tokens": do_add_special_tokens,
        "padding": True,
    }
    check_tokenizer_output(
        tiktoken_tokenizers_with_padding_options,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        *eng_test_strings,
        *multilingual_test_strings,
        *emoji_test_strings,
        *misc_strings,
    ],
)
def test_tiktoken_detokenizer(
    tiktoken_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        tiktoken_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        string.split()
        for string in (
            *eng_test_strings,
            *multilingual_test_strings,
            *emoji_test_strings,
            *misc_strings,
        )
        if string.split()
    ],
)
def test_wordlevel_tokenizers(wordlevel_tokenizers, test_string, do_add_special_tokens):
    hf_tokenizer_kwargs = {"add_special_tokens": do_add_special_tokens, "padding": True}
    check_tokenizer_output(
        wordlevel_tokenizers,
        test_string=test_string,
        skip_missing_outputs=True,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        calculate_diff=True,
    )


@pytest.mark.parametrize(
    "test_string",
    [
        string.split()
        for string in (
            *eng_test_strings,
            *multilingual_test_strings,
            *emoji_test_strings,
            *misc_strings,
        )
        if string.split()
    ],
)
def test_wordlevel_detokenizer(
    wordlevel_tokenizers_detokenizers, test_string, do_skip_special_tokens, do_clean_up_tokenization_spaces
):
    hf_detokenizer_kwargs = {
        "skip_special_tokens": do_skip_special_tokens,
        "padding": True,
        "clean_up_tokenization_spaces": do_clean_up_tokenization_spaces,
    }
    check_detokenizer_output(
        wordlevel_tokenizers_detokenizers,
        test_string=test_string,
        hf_detokenizer_kwargs=hf_detokenizer_kwargs,
    )


def test_streaming_detokenizer(sentencepiece_streaming_tokenizers):
    hf_tokenizer, _, ov_detokenizer = sentencepiece_streaming_tokenizers

    test_string = "this is a test string"
    tokenized_string = hf_tokenizer(test_string).input_ids
    hf_detokenized = hf_tokenizer.decode(tokenized_string)

    detokenized_stream = ""
    for token in tokenized_string:
        ov_output = ov_detokenizer(np.atleast_2d(token))["string_output"][0]
        detokenized_stream += ov_output

    assert detokenized_stream == hf_detokenized


def test_detokenizer_results_align_with_hf_on_multitoken_symbols_for_streaming():
    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat", trust_remote_code=True)
    _, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    ov_detokenizer = core.compile_model(ov_detokenizer)

    test_string = "🤷‍♂️"  # tokenized into 5 tokens
    tokenized_string = hf_tokenizer(test_string).input_ids

    detokenized_stream = ""
    hf_detokenized_stream = ""
    for token in tokenized_string:
        ov_output = ov_detokenizer(np.atleast_2d(token))["string_output"][0]
        detokenized_stream += ov_output

        hf_output = hf_tokenizer.decode(token)
        hf_detokenized_stream += hf_output

    assert detokenized_stream == hf_detokenized_stream


def check_rt_info(hf_tokenizer, *models: Model) -> None:
    for model in models:
        assert model.has_rt_info(ORIGINAL_TOKENIZER_CLASS_NAME), ORIGINAL_TOKENIZER_CLASS_NAME
        assert model.get_rt_info(ORIGINAL_TOKENIZER_CLASS_NAME) == str(type(hf_tokenizer))

        for field_name, attributes in rt_info_to_hf_attribute_map.items():
            attribute = get_hf_tokenizer_attribute(hf_tokenizer, attributes)
            if attribute is None:
                assert not model.has_rt_info(field_name), field_name
            else:
                assert model.has_rt_info(field_name), field_name
                assert model.get_rt_info(field_name).value == attribute, (
                    field_name,
                    attributes,
                    model.get_rt_info(field_name).value,
                )


def test_rt_info_wordpiece(hf_wordpiece_tokenizers):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_wordpiece_tokenizers,
        with_detokenizer=True,
    )
    check_rt_info(hf_wordpiece_tokenizers, ov_tokenizer, ov_detokenizer)


def test_rt_info_bpe(hf_bpe_tokenizers):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_bpe_tokenizers,
        with_detokenizer=True,
    )
    check_rt_info(hf_bpe_tokenizers, ov_tokenizer, ov_detokenizer)


def test_rt_info_tiktoken(hf_tiktoken_tokenizers):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tiktoken_tokenizers,
        with_detokenizer=True,
    )
    check_rt_info(hf_tiktoken_tokenizers, ov_tokenizer, ov_detokenizer)


def test_rt_info_sentencepiece(hf_sentencepiece_tokenizers, is_sentencepiece_backend, is_fast_tokenizer):
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_sentencepiece_tokenizers, with_detokenizer=True, use_sentencepiece_backend=is_sentencepiece_backend
    )
    check_rt_info(hf_sentencepiece_tokenizers, ov_tokenizer, ov_detokenizer)


models_to_check_rt_info = [
    # one model from each category
    "bert-base-uncased",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Xenova/gpt-4o",
    "Qwen/Qwen-14B-Chat",
]


@pytest.fixture(scope="session", params=models_to_check_rt_info)
def tokenizer_to_check_rt_info(request):
    return get_hf_tokenizer(request, trust_remote_code=True)


def test_rt_info_conversion_params(tokenizer_to_check_rt_info):
    conversion_params = TokenzierConversionParams(
        with_detokenizer=False,
        add_special_tokens=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=None,
        tokenizer_output_type=Type.i64,
        detokenizer_input_type=Type.i64,
        streaming_detokenizer=False,
        use_max_padding=False,
        handle_special_tokens_with_re=None,
        use_sentencepiece_backend=False,
        utf8_replace_mode=None,
        number_of_inputs=1,
    )

    ov_tokenizer = convert_tokenizer(tokenizer_to_check_rt_info, conversion_params)
    print(ov_tokenizer)
    if not conversion_params.with_detokenizer:
        ov_tokenizer = (ov_tokenizer,)

    for model in ov_tokenizer:
        for key in fields(conversion_params):
            val = getattr(conversion_params, key.name)
            if val is None:
                val = {}
            elif isinstance(val, (Type, int)) and not isinstance(val, bool):
                # bool is subcalss of int, hence there are 2 checks.
                # While bool values are stored as str, e.g. 'False'
                # type info and integers are stored as object.
                pass
            else:
                val = str(val)
            assert val == model.get_rt_info(key.name).value


cache_test_strings = [
    "Eng... test, string?!",
    "Multiline\nstring!\nWow!",
    "A lot\t w!",
    "A lot\t\tof whitespaces!",
    "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n",
    "Eng, but with d1gits: 123; 0987654321, stop.0987654321 - eng, but with d1gits: 123",
    "<s>[INST] <<SYS>> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. <</SYS>> You will act as a Christian, and fully summarize following text:\nSometimes it's nice to take a minute in the pew by yourself beforehand. You have this beautiful church probably almost all to yourself. Can you feel its energy resonating through you? Can you feel the majesty of the Lord's kingdom and how you're a part of it? Take a moment to kneel and pray with your head down and hands clasped together. Reflect on your faith and how you feel currently. Think about how you've been responding to God's call and how you've been living in the light of his love. When the priest is ready for you, of course. You'll probably see him there by his lonesome or someone else walk out just before you. Sit down either across from him or behind the screen -- it's totally up to you whether or not you prefer to remain anonymous. He won't treat you any differently either way. Make the sign of the cross upon his prompt, saying, \"Bless me, Father, for I have sinned. It has been 10 years since my last confession.\" This is your standard, traditional phrasing. However, if you just sit down and say hello, that's fine, too. The priest knows what he's doing. The Byzantine Rite is a bit different. The priest may sit to your side and put his epitrachelion on your head. He may then also do the Prayer of Absolution. But the idea remains the exact same -- just go wherever he takes you. Once you sit down and you've made the sign of the cross, just sit back and follow the priest's lead. He'll ask you how long it's been since your last confession (if you don't voluntarily offer that information), how you are feeling, maybe how your faith is going, and then ask you what sins you would like to talk about with him and God. It's just a casual conversation! Do not fret. There is absolutely zero pressure on your part. Again, as long as you come there with the intention of leaving with a clean heart, you're more than welcome in the church. There is no wrong way to go about confession! This part is intimidating, but think about it this way: the priest you're talking to has probably heard just about everything before. Whatever you have to say will not blow his mind. So when he asks, start rattling them off, from the most serious to the least. If he asks any questions, answer them, but do not feel the need to go into detail. A simple, \"I did so and so,\" will suffice. Your priest is going to be very understanding. If you don't remember the exact timeframe, that's fine. If you don't remember your motivation, that's fine. All your priest cares about is that you're being as honest as possible and that your heart is in the right place. He'll talk you through everything, possibly asking about your intentions, but mainly just letting you know that God loves you, sin and all. If he has any ideas to bring you closer to God, he may suggest them at this juncture. He's there to help, after all. He will then ask you to make an Act of Contrition. That goes like this: My God, I am sorry for my sins with all my heart.In choosing to do wrong and failing to do good,I have sinned against You whom I should loveabove all things. I firmly intend, with your help,to do penance, to sin no more, andto avoid whatever leads me to sin.Our Savior Jesus Christ suffered and died for us.In his name, my God, have mercy.If you are a Roman Catholic, your act of contrition will go like this: Oh my God, I am very sorry for having offended thee. But most of all, because they offend you, my God, who is all good and deserving of all my love. I firmly resolve with the help of thy grace, to sin no more, and to avoid the near occasion of sin. Amen. Don't worry! It won't be anything huge. Take the absolution to heart -- you now have a brand new, clean slate to work with. \"Penance\" is your expression of regret and repentance, showing God that you're truly sorry and that you wish for nothing more than to be forgiven. Thanks. [/INST]",
]


@pytest.mark.parametrize(
    "model_id",
    [
        "Xenova/gpt-4o",
    ],
)
@pytest.mark.parametrize("test_string", cache_test_strings)
def test_loading_from_cache(tmp_path, model_id, test_string):
    request = namedtuple("request", ["param"])(model_id)

    hf_tokenizer = get_hf_tokenizer(request, trust_remote_code=True)
    ov_tokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=False)

    save_model(ov_tokenizer, tmp_path / "openvino_tokenizer.xml")
    ov_tokenizer = Core().read_model(tmp_path / "openvino_tokenizer.xml")

    # Compile with cache dir, to check if after restoration still will work fine.
    compiled_tokenizer = Core().compile_model(ov_tokenizer, "CPU", {properties.cache_dir: str(tmp_path)})
    check_tokenizer_output((hf_tokenizer, compiled_tokenizer), test_string=test_string)

    # On the second run, it should be loaded from cache.
    # Check that output is still the same
    compiled_tokenizer = Core().compile_model(ov_tokenizer, "CPU", {properties.cache_dir: str(tmp_path)})
    check_tokenizer_output((hf_tokenizer, compiled_tokenizer), test_string=test_string)


@pytest.fixture(scope="session")
def hf_model(request):
    hf_tokenizer = get_hf_tokenizer(request, trust_remote_code=True)
    ov_tokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=False, number_of_inputs=2)
    ov_tokenizer = Core().compile_model(ov_tokenizer, "CPU")
    return hf_tokenizer, ov_tokenizer


@pytest.mark.parametrize("test_string", [   
    [["hi", "sun in yellow"]],
    [["hi", "\n\n\n\t\t   A    lot\t\tof\twhitespaces\n!\n\n\n\t\n\n"]],
    [["Eng... test, string?!", "Multiline\nstring!\nWow!"]],
    [["Eng... test, string?!" * 100, "Multiline\nstring!\nWow!"]],
    [["Eng... test, string?!", "Multiline\nstring!\nWow!" * 100]],
])
@pytest.mark.parametrize(
    "hf_model",
    [
        "answerdotai/ModernBERT-base",
        "amberoad/bert-multilingual-passage-reranking-msmarco",
        "BAAI/bge-reranker-v2.5-gemma2-lightweight",
        
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "koalajun/Gemma-2-9b-it-Ko-Crypto-Translate",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "cointegrated/rubert-tiny2",
        "google/mobilebert-uncased",
        "microsoft/deberta-base",
        
        # Rerankers with sentencepiece
        # "BAAI/bge-reranker-v2-m3",
        # "BAAI/bge-reranker-base",

        # Fail when string exceed max_length
        # "sentence-transformers/all-MiniLM-L6-v2",
        # "rasa/LaBSE",
        # "bert-base-multilingual-cased",
    ],
    indirect=True
)
def test_pair_input(hf_model, test_string):
    check_tokenizer_output(hf_model, test_string=test_string)
