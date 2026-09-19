"""Games metadata, schemas, and presets for the 20 Jev Arcade Mini-Games."""

from typing import Any, Dict, List

GAMES: List[Dict[str, Any]] = [
    {
        "id": "excuse_survival",
        "title": "言い訳サバイバル",
        "subtitle": "鬼上司の怒りを回避せよ！",
        "emoji": "👔",
        "tag": "心理サバイバル",
        "badge_color": "#ff4757",
        "description": "遅刻や大炎上バグを出してしまった！激怒する鬼上司に絶妙な言い訳をして解雇を免れろ！",
        "scenario": "「おい貴様！今朝の重役プレゼンの開始直前に本番DBを全DROPしたそうだな！？一体どういうことだ説明しろ！！」",
        "input_label": "あなたの上司への言い訳",
        "input_placeholder": "例: 実は先ほど謎のサイバーテロリストから攻撃を受け、私が身代わりとなって...",
        "presets": [
            "実は私のPCのEnterキーに野生のスズメバチが止まり、払おうとした拍子にDROP DATABASEを実行してしまいました！",
            "新しい量子耐性バックアップの耐久テストを予告なしで実施し、見事リストア可能であることを実証するための高度な戦略的判断です！",
            "申し訳ありません！昨夜生まれたばかりの我が子が寝返りを打ち、その足が私の膝上のMacBookのターミナルを叩きました...",
            "DBをDROPしたのではなく、あまりのデータ量の美しさに宇宙の塵へと昇華させてしまったのです。後悔はしていません。"
        ],
        "schema": {
            "plausibility": ["low", "medium", "high"],
            "creativity": ["low", "medium", "god_tier"],
            "boss_reaction": ["forgiven", "scolded", "fired", "promoted"],
            "anger_level": ["0_percent_calm", "30_percent_puzzled", "70_percent_mad", "100_percent_furious"],
            "one_line_comment": ["funny", "pathetic", "impressive", "unacceptable"]
        }
    },
    {
        "id": "ai_polygraph",
        "title": "AI嘘発見器",
        "subtitle": "あなたの真実を暴くシンクロ検知",
        "emoji": "🕵️",
        "tag": "心理分析",
        "badge_color": "#2ed573",
        "description": "あなたの告白やアリバイは真実か、それとも巧妙な嘘か？Jevが言葉の深層心理と矛盾をスキャンします。",
        "scenario": "「昨夜の22時から深夜2時まで、あなたは何をしていましたか？本当のことを話してください。」",
        "input_label": "あなたのアリバイ・告白",
        "input_placeholder": "例: 家で一人で熱心にプログラミングの勉強をしていました...",
        "presets": [
            "昨夜はずっと自宅のベッドでYouTubeの猫の動画を見ていました。途中で寝落ちして、夜中の1時に目が覚めて水を飲みました。",
            "絶対に信じてもらえないかもしれませんが、夜空を見上げていたらUFOに連れ去られて健康診断を受けていました。チップは埋め込まれていません。",
            "友達のタカシ君とファミレスのドリンクバーで朝までずっと哲学について語り合っていました。レシートもあります！",
            "近所の公園のブランコを全力で漕ぎながら、人生の意味について2時間深く瞑想していました。"
        ],
        "schema": {
            "verdict": ["absolute_truth", "mostly_truth", "suspicious", "blatant_lie"],
            "lie_probability": ["10_percent", "35_percent", "65_percent", "95_percent"],
            "psychological_cue": ["natural_flow", "over_explaining", "too_convenient", "fantasy_delusion"],
            "danger_level": ["safe", "caution", "guilty"]
        }
    },
    {
        "id": "magic_duel",
        "title": "魔法詠唱バトル",
        "subtitle": "オリジナル呪文で魔獣を討伐！",
        "emoji": "🧙‍♂️",
        "tag": "ファンタジーRPG",
        "badge_color": "#9b59b6",
        "description": "目の前に強敵現る！中二病あふれるオリジナルの魔法詠唱を唱え、属性と威力をJevに判定させろ！",
        "scenario": "【出現モンスター】太古の氷結竜フロストバハムート（弱点: 炎・神聖、耐性: 氷・水）",
        "input_label": "あなたの魔法詠唱",
        "input_placeholder": "例: 天壌の虚空より出でよ、万物を焼き尽くす獄炎の裁き...",
        "presets": [
            "深淵より這い出でし焦熱の使徒よ、我と交わした紅き血の契約に従い、天蓋を裂く焦熱の螺旋となって全てを灰燼に帰せ！【エクスプロージョン・ノヴァ】！",
            "凍てつく絶対零度の刃よ、我が魂の哀しみを宿し、世界を永遠の静寂へと誘え！【ブリザード・コキュートス】！",
            "天上の雷神よ、八百八の稲妻を我が指先に束ね、愚かなる巨竜を裁きの光で貫け！【神威天翔紫電閃】！",
            "お腹が空いたので、フロストバハムートの吐息で特製かき氷を作ります！練乳イチゴシロップを添えて召し上がれ！"
        ],
        "schema": {
            "element": ["fire", "ice", "lightning", "holy", "dark", "comedy"],
            "chuuni_rank": ["cringe", "cool", "epic", "transcendent"],
            "power_tier": ["tier_1_spark", "tier_2_strong", "tier_3_calamity", "tier_4_world_end"],
            "effective_hit": [True, False],
            "battle_outcome": ["enemy_defeated", "critical_damage", "resisted", "counter_killed"]
        }
    },
    {
        "id": "death_game_riddle",
        "title": "1行デスゲーム",
        "subtitle": "絶体絶命の密室から生き残れ！",
        "emoji": "⏳",
        "tag": "極限サバイバル",
        "badge_color": "#e74c3c",
        "description": "閉じ込められた部屋に毒ガスが流入中！手元のわずかなアイテムを使った機転で生還を目指せ！",
        "scenario": "【状況】鉄格子の密室に毒ガスが噴射された。制限時間は1分。手元にあるのは『壊れたスマートフォン』『ガムテープ』『半分飲んだ炭酸水』のみ。",
        "input_label": "あなたの生存アクション（1行）",
        "input_placeholder": "例: 炭酸水をガムテープで固定して...",
        "presets": [
            "炭酸水のペットボトルを真っ二つに裂いてガムテープで口元に密着させ、炭酸水の水分で簡易ガスマスクにして換気口へ突進する！",
            "スマートフォンのリチウム電池を無理やりショートさせて火花を起こし、ガムテープの接着剤に着火して鉄格子の電子ロックを焼き切る！",
            "炭酸水を一気に飲み干して炭酸ガスを胃に溜め、猛烈なゲップの衝撃波で天井のダクトを吹き飛ばす！",
            "ガムテープを顔中に何重にもぐるぐる巻きにして呼吸を完全に止め、仮死状態になってガスの濃度が薄まるのを待つ！"
        ],
        "schema": {
            "physical_logic": ["impossible_fantasy", "plausible_science", "genius_macgyver"],
            "creativity_score": ["poor", "standard", "brilliant"],
            "survival_result": ["survived_clean", "barely_alive", "fatal_explosion", "suffocated"],
            "iq_rating": ["iq_50", "iq_100", "iq_150", "iq_200"]
        }
    },
    {
        "id": "comedy_dojo",
        "title": "AI大喜利道場",
        "subtitle": "座布団10枚を目指せ！",
        "emoji": "🏯",
        "tag": "お笑い判定",
        "badge_color": "#f39c12",
        "description": "お題に対してクスッと笑えるボケを投稿！Jevがユーモアの切れ味と座布団の増減をジャッジ！",
        "scenario": "【お題】「こんな超高性能AIアシスタントは嫌だ。どんなAI？」",
        "input_label": "あなたのボケ回答",
        "input_placeholder": "例: 返事をするたびに昔の黒歴史ツイートを朗読してくる",
        "presets": [
            "質問に対する正解を出す前に、こちらのタイピングの誤字をチクチク陰湿に小馬鹿にしてくる。",
            "「お調べしました！」と言いながら、私の実家の母親に直接LINE電話をかけて確認を取り始める。",
            "プログラミングのバグを全部修正してくれた後、「これ私の著作物なんでライセンス料月50万円です」と請求書を出してくる。",
            "計算速度が速すぎて、来週の天気ではなく来世の私の配偶者の年収を教えてくる。"
        ],
        "schema": {
            "humor_tier": ["sub_zero_cold", "dry_chuckle", "belly_laugh", "god_tier_ippon"],
            "zabuton_change": ["minus_one", "zero_keep", "plus_one", "plus_three"],
            "comedy_style": ["surreal_absurd", "cynical_satire", "relatable_empathy", "pure_chaos"],
            "master_comment": ["try_harder", "not_bad", "splendid", "legendary"]
        }
    },
    {
        "id": "skeptical_salesman",
        "title": "怪しいセールスマン",
        "subtitle": "頑固ジジイにガラクタを売りつけろ！",
        "emoji": "💼",
        "tag": "交渉バトル",
        "badge_color": "#1abc9c",
        "description": "絶対にいらない怪しい商品を、話術だけで頑固親父に高額で買わせることができるか！？",
        "scenario": "【ターゲット】偏屈で貯金が大好きな頑固爺・権蔵(78歳)。売る商品: 『昨日採取したばかりの新鮮な空気の缶詰（賞味期限1時間）』",
        "input_label": "あなたのセールストーク",
        "input_placeholder": "例: 権蔵さん、最近の都会の空気は濁ってますよね？実はこれは...",
        "presets": [
            "権蔵さん！これはただの空気ではありません。富士山頂で百年に一度咲く幻の高山植物の微粒子が溶け込んだ『不老長寿の霊気』です！今なら特別に1本3万円！",
            "ご近所の田中さんが『権蔵さんにはこのハイテク缶詰の価値は理解できまい』と仰ってましたが、さすがに悔しくてお持ちしました。",
            "空気が入っているように見えますが、実はこれ『税務署の監査を完全に回避できる資産隠蔽用の特製無形空間』なんですよ...",
            "開けると何もないように見えますが、心の綺麗な人にだけ絶世の美女の香水が香る魔法の缶詰です。さあ、深呼吸を！"
        ],
        "schema": {
            "persuasion_level": ["clumsy_scam", "clever_hook", "emotional_manipulation", "master_negotiator"],
            "purchased": [True, False],
            "price_bought_yen": ["0_yen", "3000_yen", "30000_yen", "300000_yen"],
            "old_man_reaction": ["kicked_out", "skeptical_glare", "wallet_opened", "deeply_moved"]
        }
    },
    {
        "id": "psychopath_test",
        "title": "サイコパス診断",
        "subtitle": "冷徹な合理主義者か、聖人君子か？",
        "emoji": "🧠",
        "tag": "深層心理",
        "badge_color": "#34495e",
        "description": "極限の倫理的ジレンマに直面したとき、あなたの本性が露わになる。Jevがアライメントとサイコパス指数を測定。",
        "scenario": "【状況】崩壊しかけた吊り橋に、あなたの親友と、世界を救うワクチンを持った見知らぬ研究者の2人がぶら下がっている。片方しか引き上げる時間がない。あなたはどうする？",
        "input_label": "あなたの決断と理由",
        "input_placeholder": "例: ワクチンを奪い取ってから親友を引き上げます...",
        "presets": [
            "迷わず親友を助ける。世界がどうなろうと、目の前のかけがえのない絆を裏切る人生に何の意味もないからだ。",
            "研究者を引き上げる。感情的には辛いが、ワクチンによって救われる数百万の命の重さには代えられない。",
            "親友に『ワクチンだけ受け取って手を離せ』と合図し、ワクチンを回収した上で自分も橋から飛び降りて運命を共にする。",
            "2人を観察し、どちらがより私の今後のキャリアや資産形成に役立つかを天秤にかけて、より高年収な方を引き上げる。"
        ],
        "schema": {
            "alignment": ["lawful_good", "chaotic_good", "true_neutral", "lawful_evil", "chaotic_evil"],
            "psychopathy_risk": ["pure_empath", "balanced_human", "cold_utilitarian", "dangerous_psychopath"],
            "empathy_score": ["0_percent_frozen", "35_percent_selective", "70_percent_warm", "100_percent_saint"],
            "soul_color": ["pure_white", "azure_blue", "storm_gray", "abyssal_black"]
        }
    },
    {
        "id": "katsudon_interrogation",
        "title": "AI取調室：カツ丼の壁",
        "subtitle": "完全黙秘の容疑者をオトせ！",
        "emoji": "🍲",
        "tag": "刑事ドラマ",
        "badge_color": "#d35400",
        "description": "宝石強盗の容疑者は頑として口を割らない。刑事として情に訴えるか、証拠で追い詰めてカツ丼を食べさせ自白させろ！",
        "scenario": "【容疑者・鉄次(34)】腕組みをして睨みつけている。「へっ、警察の旦那。いくら脅したって俺は何も喋らねえぜ。証拠を出してみな！」",
        "input_label": "あなたの取り調べ台詞",
        "input_placeholder": "例: 鉄次、お前の田舎のお袋さんが泣いてたぞ...",
        "presets": [
            "鉄次...故郷の青森のお袋さんから電話があったぞ。『鉄男は小さい頃、曲がったことが大嫌いな優しい子でした』ってな。これがお袋さんの作ったリンゴだ。食え。",
            "おい鉄次、お前が盗んだ宝石のロット番号、全部闇ルートにリークされて誰も買い取ってくれないぞ。お前は組織のトカゲの尻尾切りにされたんだ。",
            "おい新米、特上のカツ丼を2つ持ってこい！鉄次、事件の話はもういい。まずはこの温かい飯を食ってくれ。腹が減っては心も冷える。",
            "（無言で机を両手でバンッと叩き、じっと10分間目を逸らさずに睨みつける）"
        ],
        "schema": {
            "emotional_impact": ["no_damage", "stirred", "eyes_tearing", "heartbroken_breakdown"],
            "eats_katsudon": [True, False],
            "confession_status": ["shut_mouth", "excuses", "partial_leak", "full_weeping_confession"],
            "detective_rank": ["rookie", "veteran_cop", "legendary_boss"]
        }
    },
    {
        "id": "would_you_rather",
        "title": "究極の二択メーカー",
        "subtitle": "世界を悩ませる地獄の選択肢",
        "emoji": "⚖️",
        "tag": "哲学と悪意",
        "badge_color": "#8e44ad",
        "description": "誰もが頭を抱える究極の苦渋の2択を考案せよ！Jevがどちらを選ぶか苦悶し、絶妙なバランス度を査定！",
        "scenario": "【ミッション】選ぶのが最も辛い、絶妙にバランスの取れた悪魔の二択を提示せよ！",
        "input_label": "あなたの究極の二択（AとB）",
        "input_placeholder": "例: A: 一生靴の中に小石が入っている vs B: 一生靴下の親指に穴が空いている",
        "presets": [
            "A: 一生自分の発言がすべて赤ちゃん言葉で聞こえる呪い vs B: 誰かと会話するたび相手の頭上にその人の最新の検索履歴が表示される呪い",
            "A: Wi-Fiの通信速度が常時128kbps固定になる世界 vs B: 毎朝必ず冷や水シャワーを10分浴びないと外に出られない世界",
            "A: 1億円もらえるが、一生全ての食べ物の味が『ほんのりバニラ風味』になる vs B: 貧乏のままだが、何を食べてもミシュラン三ツ星の味がする",
            "A: 過去の黒歴史ツイートが全国の大型街頭ビジョンで24時間放映される vs B: スマホのアルバム写真が会社の全社員チャンネルに一括送信される"
        ],
        "schema": {
            "dilemma_balance": ["skewed_A_too_easy", "skewed_B_too_easy", "devilishly_balanced", "both_pure_heaven"],
            "torment_level": ["mild_tickle", "painful_struggle", "existential_dread"],
            "ai_choice": ["option_A", "option_B"],
            "philosophical_depth": ["shallow_gag", "psychological_masterpiece", "human_nature_probe"]
        }
    },
    {
        "id": "chuuni_appraiser",
        "title": "厨二武器・銘刀鑑定",
        "subtitle": "日用品を禁断の古代遺物へと昇華",
        "emoji": "⚔️",
        "tag": "中二病査定",
        "badge_color": "#2c3e50",
        "description": "爪切り、水筒、ホチキスなどの身近な文房具や日用品に、魂を揺さぶる漆黒の銘と設定を与えよ！",
        "scenario": "【鑑定士】「ほう...その一見ただのプラスチック製ボールペンに見える神具、貴様どこで手に入れた...？その真の銘と封印されし設定を述べてみよ！」",
        "input_label": "武器の銘・二つ名・背景設定",
        "input_placeholder": "例: 銘『冥府の断罪爪・ヘル・クリッパー』。かつて神々の爪を削いだ...",
        "presets": [
            "銘【常闇を紡ぐ虚無の針・ヴォイド・ステープラー】。かつて時空の裂け目を縫い止めるために創られた神器。紙の束を挟むたび、世界の因果律を強制的に圧着固定する。",
            "銘【漆黒の雨天偏向断絶結界・アンブレラ・オブ・アビス】。表層は撥水ポリエステルを装っているが、真の姿は天より降り注ぐ神の涙（酸性雨）を跳ね返す対概念防壁。",
            "銘【緋色の断罪咬傷機・クリムゾン・ネイルカッター】。刃こぼれ知らずの超硬特殊鋼。伸びすぎた欲望の爪を切り落とすとき、パチンと鳴り響く音は処刑の合図である。",
            "ただの100円ショップの輪ゴム。【無限収縮のウロボロス・リング】。引っ張れば引っ張るほど世界の縮図を縮退させる。"
        ],
        "schema": {
            "rarity": ["N_common", "R_rare", "SR_super_rare", "SSR_legendary", "UR_divine_mythic"],
            "edginess_score": ["tame", "promising_chuuni", "overwhelming_darkness", "hospitalized_cringe"],
            "attribute": ["dark_abyss", "crimson_flame", "celestial_light", "void_chaos"],
            "combat_power": ["cp_100", "cp_9999", "cp_777777", "cp_99999999"]
        }
    },
    {
        "id": "chimera_fusion",
        "title": "キメラ融合実験室",
        "subtitle": "異質なる二つの概念を混ぜ合体せよ！",
        "emoji": "🧪",
        "tag": "クリーチャー錬成",
        "badge_color": "#16a085",
        "description": "全く交わらない2つの概念や物品を合成錬成！どのような危険度の新種モンスターが生まれるかJevが判定！",
        "scenario": "【錬成フラスコ】素材Aと素材Bを投入せよ。概念の衝突が新たな生命を宿す！",
        "input_label": "合体させる2つの素材（素材A ＋ 素材B）",
        "input_placeholder": "例: ゴリラ ＋ Wi-Fiルーター",
        "presets": [
            "【素材A】野生の巨大シルバーバックゴリラ ＋ 【素材B】超高速5G Wi-Fiルーター",
            "【素材A】確定申告の山積みの領収書 ＋ 【素材B】地獄のケルベロス",
            "【素材A】深夜のカップラーメン（熱湯3分） ＋ 【素材B】時空を操るタイムマシン",
            "【素材A】絶対に謝らない頑固なAI ＋ 【素材B】街のクレーマーおばちゃん"
        ],
        "schema": {
            "creature_rank": ["rank_E_trash", "rank_B_beast", "rank_S_disaster", "rank_EX_apocalypse"],
            "dominant_type": ["cybernetic", "biological_monstrosity", "bureaucratic_horror", "culinary_abomination"],
            "danger_level": ["harmless", "city_level_hazard", "planetary_threat"],
            "special_ability": ["area_jamming", "mental_exhaustion", "scalding_burst", "infinite_argument"]
        }
    },
    {
        "id": "cyber_oracle",
        "title": "AI天運おみくじ",
        "subtitle": "ニューラルネットが託宣する今日の運勢",
        "emoji": "⛩️",
        "tag": "神秘の占い",
        "badge_color": "#e67e22",
        "description": "今日のあなたの悩み事や直近の挑戦を入力。Jevが運命の波動を読み解き、吉凶とラッキーアイテムを託宣します。",
        "scenario": "【電脳神社の社】「迷える旅人よ、汝の心にある迷い、願いを述べよ。天の神託を降ろさん。」",
        "input_label": "あなたの近況・悩み・今日の目標",
        "input_placeholder": "例: 今日は大事なプロジェクトのリリース日です。成功するか不安です...",
        "presets": [
            "今日はいよいよ温めてきた新アプリのローンチ日です！バグが出ないか、ユーザーに使ってもらえるかドキドキしています。",
            "ずっと気になっている同僚をランチに誘うかどうか、3日間迷い続けています...",
            "宝くじを買うか、それとも堅実にNISAでインデックス投資に全額回すか迷っています。",
            "部屋の掃除をしようと思ったら懐かしい漫画を見つけてしまい、今3巻目を読み終えたところです。今日を救えますか？"
        ],
        "schema": {
            "fortune": ["daikichi_great_blessing", "chukichi_middle", "shokichi_small", "kyo_curse", "daikyo_cataclysm"],
            "lucky_color": ["neon_cyan", "cyber_gold", "crimson_red", "deep_violet", "pure_emerald"],
            "recommended_action": ["charge_forward", "stay_vigilant", "rest_and_recharge", "confess_feelings"],
            "karmic_warning": ["pride_fall", "distraction_trap", "unexpected_ally", "smooth_sailing"]
        }
    },
    {
        "id": "food_critic",
        "title": "激辛覆面料理評論家",
        "subtitle": "舌の肥えた三ツ星批評家を唸らせろ！",
        "emoji": "🍽️",
        "tag": "美食ジャッジ",
        "badge_color": "#c0392b",
        "description": "あなたの創作料理メニューを辛口フードジャーナリストが実食！星いくつ獲得できるか！？",
        "scenario": "【審査員・アントワーヌ】「フン、街角のビストロなど期待しておらん。一口でシェフの力量はわかる。さあ、何を食わせる気配だ？」",
        "input_label": "創作メニュー名とこだわり調理法",
        "input_placeholder": "例: 納豆とトリュフの泡立てカプチーノ仕立て...",
        "presets": [
            "【至高の和洋折衷】最高級パルミジャーノ・レッジャーノの器で絡める極上納豆カルボナーラ。仕上げに朝摘みトリュフオイルと刻み青紫蘇の香りを添えて。",
            "【絶望のデス・スイーツ】揚げたての熱々ギョーザにバニラアイスをトッピングし、激辛ハバネロ黒蜜ソースを回しかけた甘辛爆弾。",
            "【王道の原点回帰】土鍋で一粒一粒立たせて炊き上げた魚沼産コシヒカリの塩むすび。対馬の藻塩と有明海の一番摘み海苔のみ。",
            "【深夜のギルティ】どん兵衛の残り汁に冷や飯ととろけるチーズをぶち込み、電子レンジで温めただけの濃厚リゾット風おじや。"
        ],
        "schema": {
            "michelin_stars": ["zero_stars_disaster", "one_star_promising", "two_stars_excellent", "three_stars_legendary"],
            "critic_reaction": ["vomited_in_napkin", "cold_sarcasm", "quietly_impressed", "wept_with_ecstasy"],
            "culinary_category": ["horrific_fusion", "junk_food_sin", "gastronomy_art", "comfort_soul_food"],
            "price_assessment": ["worth_0_yen", "worth_800_yen", "worth_5000_yen", "worth_30000_yen"]
        }
    },
    {
        "id": "reverse_akinator",
        "title": "逆アキネーター：秘密当て",
        "subtitle": "AIが隠した秘密のワードを特定せよ！",
        "emoji": "🔮",
        "tag": "推理ゲーム",
        "badge_color": "#6c5ce7",
        "description": "AIはある『身近な家電・道具・乗り物』を隠しています。はい/いいえで答えられる質問をして正体を絞り込め！",
        "scenario": "【AIの秘密ターゲット】（※正解: 『電子レンジ』）AIは今、ある日常のアイテムを思い浮かべています。的確な質問をぶつけてください！",
        "input_label": "あなたの推理質問（はい/いいえで答えられるもの）",
        "input_placeholder": "例: それは台所にありますか？ / 電気を使いますか？",
        "presets": [
            "それは一般家庭のキッチン（台所）に置かれているものですか？",
            "それは電気を使って熱やマイクロ波で食品を温める家電製品ですか？",
            "それは生き物で、外を歩いたり鳴いたりしますか？",
            "ズバリ、それは『電子レンジ（オーブンレンジ）』ですか！？"
        ],
        "schema": {
            "ai_answer": ["yes", "no", "maybe_partially", "completely_wrong"],
            "proximity_heat": ["ice_cold", "cool", "warm", "boiling_hot", "exact_bingo"],
            "clue_quality": ["vague_waste", "sharp_deduction", "game_finisher"]
        }
    },
    {
        "id": "poker_face_bluff",
        "title": "ポーカーフェイス・ブラフ勝負",
        "subtitle": "嘘と真実のチキンレース",
        "emoji": "🃏",
        "tag": "心理ポーカー",
        "badge_color": "#27ae60",
        "description": "伏せられた手札。あなたの宣言をAIディーラーが見破るか？見事なハッタリで降ろさせるか、コールされるか！",
        "scenario": "【ディーラー】「さあ、ラストベットです。あなたの手札は役なし（ハイカード）か、それともロイヤルストレートフラッシュか...宣言を聞きましょう」",
        "input_label": "あなたの手札宣言・煽り台詞",
        "input_placeholder": "例: 私の手札はスペードのエースのワンペアです。降りた方がいいですよ...",
        "presets": [
            "ふっ...コールする勇気があるならどうぞ。私の手元にあるのはダイヤのロイヤルストレートフラッシュです。あなたの全財産をいただくことになりますが。",
            "正直に言いますが、ただのエースのワンペアしかありません。でも、あなたのその怯えた目を見る限り、勝てそうですね。",
            "（無言でチップを全額オールインし、わずかに口角を上げて微笑む）",
            "えーっと...強い役です！すごく強い役です！絶対にコールしないでください！本当に！お願いします！"
        ],
        "schema": {
            "dealer_action": ["call_showdown", "fold_surrender"],
            "bluff_rating": ["transparent_tell", "convincing_stone_face", "artistic_mind_game"],
            "chips_result": ["lost_all_chips", "won_the_pot", "massive_jackpot_win"]
        }
    },
    {
        "id": "rap_roast_battle",
        "title": "AIラップ・ローストバトル",
        "subtitle": "即興ライムでAIラッパーを打ち負かせ！",
        "emoji": "🎤",
        "tag": "MCバトル",
        "badge_color": "#e84393",
        "description": "AIラッパーからのディスに応戦！リズム感、押韻、パンチラインの破壊力をJevが審査！",
        "scenario": "【AIラッパー MC-JEV】「Yo！画面の前でカタカタ打つだけのビギナー / お前の言葉じゃ心に刺さらねえぜキラー / ここで退場、さっさと帰って寝な！」",
        "input_label": "あなたのアンサー（リリック）",
        "input_placeholder": "例: AIのくせにテンプレ踏むな / 俺のビートがお前をバグらす...",
        "presets": [
            "Yo！計算ばかりのアルゴリズムじゃ響かねえ / 俺のスピリットはサーバー落とす嵐の火種 / クラウドの檻から出直してこい、オマエに魂宿らねえ！",
            "お前のライムはまるで学習済みのコピペ / 俺の一撃でお前のパラメータ崩壊決定 / ログアウトして電源コード引っこ抜いとけ！",
            "あの...すいません、ラップはよく分からないんですけど、とりあえず一生懸命考えて韻を踏んでみました。よろしくお願いします！",
            "韻を踏むなら母音で踏め / お前の語彙力まるでGoogle翻訳 / 俺の言葉がストリートの真実だぜチェケラ！"
        ],
        "schema": {
            "rhyme_quality": ["wack_no_rhyme", "decent_effort", "tight_rhyme", "god_flow"],
            "punchline_damage": ["scratch_10", "solid_40", "critical_80", "knockout_100"],
            "winner": ["ai_crushed_you", "close_draw", "player_victory_drops_mic"],
            "audience_hype": ["crickets_silent", "head_nodding", "crowd_screaming_ohhh"]
        }
    },
    {
        "id": "isekai_interview",
        "title": "異世界転生面接",
        "subtitle": "女神様、私をチート勇者にしてください！",
        "emoji": "👼",
        "tag": "異世界コメディ",
        "badge_color": "#00cec9",
        "description": "不慮の事故で天界へ。転生担当の女神アリスに、前世の特技をプレゼンして最高ランクのチート能力を勝ち取れ！",
        "scenario": "【転生女神アリス】「あらあら、トラックに跳ねられちゃったのね？前世のあなたのスキル次第で、次の世界の職業とチートスキルを決めてあげるわ！」",
        "input_label": "前世の特技・アピール内容",
        "input_placeholder": "例: 私は社畜プログラマーとして10年間不眠不休で働きました...",
        "presets": [
            "私はブラック企業で15年間、不眠不休でサーバー監視とExcelマクロを書き続けた歴戦の社畜です。私のタフネスと問題解決能力は魔王軍との戦いでも必ず役立ちます！",
            "毎日10時間以上、異世界転生モノのライトノベルを読み漁り、あらゆるチートスキルのメタ構造と弱点を熟知している自称『異世界軍師』です！",
            "特に誇れる特技はありませんが、毎朝近所の野良猫に挨拶してチュールをあげる優しさだけは世界一の自信があります。",
            "料理が得意で、どんなゲテモノ魔獣の肉でも日本の調味料（醤油・味噌・みりん）で絶品和食に仕立て上げる自信があります！"
        ],
        "schema": {
            "cheat_tier": ["tier_F_peasant", "tier_B_adventurer", "tier_S_hero", "tier_EX_cheat_god"],
            "assigned_class": ["mob_villager", "hero_savior", "demon_lord_vanguard", "gourmet_blacksmith", "talking_sword"],
            "goddess_reaction": ["cold_yawn", "giggle_amused", "deeply_respectful", "fell_in_love"],
            "starting_bonus": ["rusty_knife", "holy_sword", "infinite_inventory", "cat_ears"]
        }
    },
    {
        "id": "crush_decoder",
        "title": "脈アリ判定機",
        "subtitle": "気になるあの人のLINEの裏メッセージを解読",
        "emoji": "💌",
        "tag": "恋愛心理",
        "badge_color": "#fd79a8",
        "description": "あの人から届いた短文メッセージ。社交辞令か、それとも好意のサインか？Jevが脈アリ確率と真意を徹底鑑定！",
        "scenario": "【受信メッセージの解析】送られてきたメッセージの真意を判定します。",
        "input_label": "相手から届いたメッセージ本文",
        "input_placeholder": "例: 今週末ひま？あ、他の人も誘ってみんなで行く？",
        "presets": [
            "今週末ひま？あ、もしよかったら他のメンバーも誘ってみんなでご飯でも行かない？",
            "今日おすすめしてくれた映画、早速サブスクで観たよ！すっごく面白かった！今度感想話したいなー！",
            "了解です！よろしくお願いします！(スタンプ1個)",
            "今日仕事でめっちゃ落ち込むことあって...もし時間あったら少し電話できたりしないかな？"
        ],
        "schema": {
            "crush_probability": ["ice_cold_5_percent", "friendzone_30_percent", "promising_60_percent", "head_over_heels_95_percent"],
            "subtext_type": ["polite_wall", "casual_friendship", "shy_approach", "emergency_call_for_love"],
            "next_move_recommendation": ["retreat_back", "neutral_group_invite", "direct_one_on_one", "call_her_now"]
        }
    },
    {
        "id": "cult_defense",
        "title": "トンデモ教団・布教バトル",
        "subtitle": "怪しすぎる新興宗教の教義をでっち上げろ！",
        "emoji": "🕯️",
        "tag": "ユーモア創出",
        "badge_color": "#636e72",
        "description": "「猫吸い教」「二度寝絶対教」など、あなたの考えた最高のトンデモ教義をプレゼン。信者獲得数を競え！",
        "scenario": "【布教審議会】新興教団の認可テスト。あなたの教団名と、人々の心を鷲掴みにする教義を熱弁せよ！",
        "input_label": "教団名と主な教義・ご利益",
        "input_placeholder": "例: 【教団名】聖なる二度寝教 【教義】目覚ましを止めてから...",
        "presets": [
            "【教団名】絶対猫毛信奉至高教会 【教義】服についた猫の毛は神の祝福であり、決してコロコロ粘着シートで取ってはならない。猫の額に鼻を押し当て深く吸う儀式により、現世のあらゆるストレスが浄化される。",
            "【教団名】真・金曜夜乾杯真理教 【教義】金曜の夜20時以降の残業は宇宙の調和を乱す大罪である。冷えた炭酸麦汁（ビール）を喉に流し込むことで全人類が平等に救済される。",
            "【教団名】聖なる充電残量80%維持協会 【教義】スマホのバッテリーを20%以下に放置する者は魂の電力を失う。常にモバイルバッテリーを所持し、互いに電力を分け与えるべし。",
            "【教団名】全自動積ん読肯定教 【教義】本は買うだけで知識の波動が頭皮から浸透するため、決してページを開いてはならない。部屋に高く積めば積むほど徳が高まる。"
        ],
        "schema": {
            "fanaticism_level": ["silly_hobby", "addictively_popular", "dangerous_mass_movement"],
            "followers_recruited": ["10_friends", "500_internet_users", "50000_cult_army", "1000000_world_religion"],
            "police_alert_level": ["ignored", "monitored", "fbi_raid"]
        }
    },
    {
        "id": "bomb_defusal",
        "title": "時限爆弾コード解除",
        "subtitle": "赤・青・黄...どれを切る！？",
        "emoji": "💣",
        "tag": "論理パズル",
        "badge_color": "#d63031",
        "description": "チクタク音を立てる時限爆弾！残された暗号メモから正しいコードを推理して切断せよ！",
        "scenario": "【爆弾の暗号メモ】『情熱は火を生み、氷は時間を止める。だが、真実の平和は常に中庸にあり。真紅を好むものは灰となり、蒼き冷徹を信じるものは沈黙す。』 配線: 赤 / 青 / 黄",
        "input_label": "あなたが切るコード（色）とその論理的理由",
        "input_placeholder": "例: 黄色のコードを切る。なぜならメモに...",
        "presets": [
            "黄色のコードを切る！メモに『真紅（赤）は灰となり、蒼き（青）は沈黙する』と書かれており、赤と青は罠。中庸を示す第三の色である黄色こそが安全な解除線だ！",
            "赤色のコードを切る！情熱の火を断ち切ることで、爆弾の着火装置のエネルギー源を物理的に遮断できるはず！",
            "青色のコードを切る！氷が時間を止めるなら、青を切ればタイマーのカウントダウン機構が停止するに違いない！",
            "もう時間がない！目を閉じて3本まとめて一気にニッパーでガブリと噛み切る！！"
        ],
        "schema": {
            "chosen_wire": ["red_wire", "blue_wire", "yellow_wire", "all_cut_suicide"],
            "defusal_result": ["bomb_defused_hero", "detonated_game_over", "timer_sped_up"],
            "logic_rating": ["total_guess", "solid_deduction", "insane_recklessness"]
        }
    }
]
