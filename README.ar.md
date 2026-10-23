<div dir="ltr">

<p align="center">
  <img src="https://raw.githubusercontent.com/Generated-Computers/Locals/8404f6b4b32b8070facd0290d157e3ed97777623/docs/assets/logo.png#gh-light-mode-only" alt="Generated Computers" width="80%">
  <img src="https://raw.githubusercontent.com/Generated-Computers/Locals/8404f6b4b32b8070facd0290d157e3ed97777623/docs/assets/logo-dark.png#gh-dark-mode-only" alt="Generated Computers" width="80%">
</p>

<h1 align="center">Locals</h1>

<p align="center">
  <a href="README.md">English</a> · <b>العربية</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/locals/"><img src="https://img.shields.io/pypi/v/locals.svg" alt="PyPI"></a>
  <a href="https://github.com/Generated-Computers/Locals/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="الترخيص: MIT"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/macOS-Apple%20silicon-lightgrey.svg" alt="macOS على معالجات Apple silicon"></a>
</p>

</div>

<div dir="rtl" lang="ar">

<p align="center">
  <b>تشغيل عدة وكلاء ذكاء اصطناعي في آنٍ واحد على جهاز Mac الخاص بك، بسرعة تصل إلى 3 أضعاف.</b><br>
  خادم محلي صُمم خصيصاً لوكلاء الذكاء الاصطناعي بدلاً من المحادثات البسيطة: استدعاء الأدوات (tool calls)، والمحادثات الطويلة، وتشغيل عدة وكلاء يعملون في الوقت نفسه. مخصص لتطبيقات OpenCode و OpenHands و Cline و Aider و Claude Code وأي تطبيق يدعم واجهات برمجة تطبيقات OpenAI أو Anthropic أو Ollama.
</p>

<p align="center">
  <a href="#getting-started"><b>البدء والاستخدام</b></a> ·
  <a href="#works-with-your-agents"><b>الوكلاء</b></a> ·
  <a href="https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md"><b>الاختبارات القياسية</b></a> ·
  <a href="#documentation"><b>التوثيق</b></a> ·
  <a href="https://arxiv.org/abs/2610.09139"><b>الورقة البحثية</b></a>
</p>

</div>

```bash
curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
```

<div dir="rtl" lang="ar">

أو إذا كنت تستخدم [uv](https://docs.astral.sh/uv/)، يمكنك تثبيته من [PyPI](https://pypi.org/project/locals/):

</div>

```bash
uv tool install --python 3.12 locals
```

<div dir="rtl" lang="ar">

ثم في نافذة طرفية (Terminal) جديدة:

</div>

```bash
locals code ~/my-project
```

<div dir="rtl" lang="ar">

هذا كل شيء. في المرة الأولى، يقوم Locals بتنزيل نموذج يلائم مواصفات جهاز Mac الخاص بك (بحجم 3 جيجابايت تقريباً)، ويبدأ تشغيله في الخلفية ثم يفتح المساعد البرمجي OpenCode داخل مشروعك. لا توجد أي بيانات تغادر حاسوبك على الإطلاق.

<a id="what-it-is"></a>
## ما هو مشروع Locals؟

<div dir="ltr">
<img src="https://raw.githubusercontent.com/Generated-Computers/Locals/8404f6b4b32b8070facd0290d157e3ed97777623/docs/assets/emblem.png" alt="Generated Computers Co." width="35%" align="right">
</div>

يقوم Locals بتشغيل نماذج الذكاء الاصطناعي مباشرة على جهاز Mac الخاص بك لخدمة وكلاء الذكاء الاصطناعي، وليس لمجرد المحادثة مع نموذج خام. إذ يعمل الوكيل بأسلوب يختلف جذرياً عن المستخدم العادي في الدردشة؛ فهو يستدعي الأدوات (tools)، وكل استدعاء يجب أن يكون دقيقاً وخالياً من الأخطاء. كما يرسل مطالبات (prompts) طويلة تنمو تدريجياً مع كل دورة محادثة. ونادراً ما يعمل الوكيل بمفرده: فالمساعدات البرمجية تُشغّل مساعدين بالتوازي، وفرق الوكلاء تتقاسم المهمة الواحدة فيما بينها. صُمم Locals خصيصاً لهذه المهام، بدءاً من طريقة معالجة دفعات الطلبات المتزامنة على معالج الرسوميات (GPU) في جهازك وحتى صياغة استدعاءات الأدوات، لكي تحصل فرق الوكلاء على إجاباتها بسرعة تفوق الطرق التقليدية لتشغيل النماذج على Mac بما يصل إلى ثلاثة أضعاف.

يعمل مع وكلاء البرمجة الذين قد تكون تستخدمهم بالفعل، مثل OpenCode و Cline و Aider و OpenHands و Claude Code، ومع أي تطبيق أو برنامج يتواصل عبر واجهات OpenAI أو Anthropic أو Ollama. كل شيء يبقى محلياً بالكامل على جهازك.

بُني Locals استناداً إلى الورقة البحثية:
[Breaking the Space Barrier and its Application to Language Model Inference](https://arxiv.org/abs/2610.09139)
(Arip Asadulaev, 2026).
حيث يمكن تمثيل صيغة JSON أو استدعاء الأداة كرسم بياني (graph)، والتحقق من صحة الإجابة بمثابة إيجاد مسار داخل هذا الرسم. تحقق الورقة البحثية ذلك باستهلاك ذاكرة أقل من حد سافيتش الكلاسيكي (Savitch's classical bound)، وتحوله إلى محرك استدلال فائق السرعة: فالنصوص التي تحددها الصيغة مسبقاً تُكتب دون الحاجة لتشغيل النموذج، وتُجرى عمليات الفحص والتحقق مباشرة على معالج الرسوميات، ولا تتطلب الصيغة سوى حوالي 3 ميجابايت من الذاكرة بدلاً من 1.5 جيجابايت. وبفضل ذلك، تنهي 16 عملية وكيل تستدعي أدوات مهامها أسرع بمقدار 2.5 مرة مقارنة بـ mlx-lm و llguidance.

<a id="made-for-agents"></a>
## مُصمم خصيصاً للوكلاء

- **عدة وكلاء في آنٍ واحد:** تُحسب طلبات جميع الوكلاء معاً، عبر نويات GPU مخصصة للتعامل مع دفعات من الطلبات المتزامنة، بحيث ينهي 8 وكلاء يعملون في وقت واحد مهامهم في نفس الوقت الذي يحتاجه وكيلان يعملان بالتتابع تقريباً.
- **استدعاءات أدوات قابلة للتحليل دائماً:** تُكتب الردود بصيغة استدعاء الأدوات الأصلية الخاصة بكل نموذج وفق قواعد نحوية (grammar) تُطبّق على معالج الرسوميات (GPU)، بحيث لا يمكن للوكيل استدعاء أدوات سوى المعروضة عليه فقط، وتصل كافة المعاملات بالنوع المحدد في مخططها (schema)، وحتى لو قُطع الرد بسبب حد الطول، فإنه يُغلق الاستدعاء بالشكل الصحيح. كما تلتزم طلبات JSON بمخططها بنفس الدقة.
- **استئناف كل دور من حيث انتهى السابق:** يحتفظ Locals بكافة المحادثات بين الأدوار، ويخزن المطالبة التوجيهية الطويلة للنظام التي تبدأ بها أدوات الوكلاء (من 3,000 إلى 16,000 رمز/token) على القرص عبر عمليات إعادة التشغيل؛ وبذلك لا يقرأ الوكيل في الدور التالي سوى ما تمت إضافته فقط: يبدأ الدور التالي لمحادثة بحجم 138,000 رمز بعد 2.4 ثانية فقط.
- **حواجز أمان مخصصة للوكلاء:** قد تكرر النماذج الصغيرة نفسها أو تستدعي نفس الأداة إلى ما لا نهاية؛ يكتشف Locals ذلك ويوقفه فوراً. كما يحمي حاجز أمان الذاكرة الوكلاء المتعددين ذوي المطالبات الطويلة من التعرض لخطأ نفاد الذاكرة (out-of-memory)، ويحرر الخادم فوراً عند إغلاق أي وكيل للاتصال.
- **أمر واحد لكل أداة وكيل:** الأمر `locals code` يفتح OpenCode في مشروعك، والأمر `locals launch` يربط كلاً من Hermes Agent و OpenClaw و Claude Code و Kilo Code و Cline و OpenHands و Aider.
- **فريق من الوكلاء للمهام الكبيرة:** الأمر `locals run` يقسم المهمة إلى أجزاء، ويسند كل جزء إلى عدة وكلاء في وقت واحد، كل وكيل في نسخته المستقلة من مشروعك، وتكون اختباراتك البرمجية هي الحكم الفاصل.

<a id="results-at-a-glance"></a>
## النتائج بلمحة سريعة

<div dir="ltr">
<img align="right" width="48%" alt="Qwen3.5-4B on an M2 Pro with 16 GB. mlx-lm: 61, 82, 87, 87, 157 tokens per second with 1, 2, 4, 8 and 16 requests at once; Locals: 64, 116, 160, 271, 341 (1.0x, 1.4x, 1.8x, 3.1x, 2.2x)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/requests-narrow.svg#gh-light-mode-only">
<img align="right" width="48%" alt="Qwen3.5-4B on an M2 Pro with 16 GB. mlx-lm: 61, 82, 87, 87, 157 tokens per second with 1, 2, 4, 8 and 16 requests at once; Locals: 64, 116, 160, 271, 341 (1.0x, 1.4x, 1.8x, 3.1x, 2.2x)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/requests-narrow-dark.svg#gh-dark-mode-only">
</div>

تقوم مكتبة mlx-lm (الطريقة القياسية لتشغيل النماذج على Mac) بالإجابة على عدة طلبات معاً، ولكنها لا تزداد سرعة عند الانتقال من 4 إلى 8 طلبات: فهي تكتب 87 رمزاً في الثانية لـ 4 طلبات ونفس الـ 87 لـ 8 طلبات. في المقابل، يواصل Locals زيادة سرعته مع إضافة الطلبات، ليصل إلى ثلاثة أضعاف سرعة mlx-lm عند وجود 8 طلبات متزامنة.

وهو أسرع أيضاً للوكيل الفردي: عندما يكرر الرد ملفاً يعيد كتابته، وبعد إعادة التشغيل، وفي المحادثات الطويلة. يوضح الجدول التالي هذه الحالات على جهاز MacBook Pro بمعالج M2 Pro وذاكرة 16 جيجابايت مع نموذج Qwen3.5-4B الافتراضي.

<div dir="ltr">
<br clear="right">
</div>

| On a MacBook Pro with an M2 Pro and 16 GB, Qwen3.5-4B | Without Locals | With Locals | |
|---|---|---|---|
| 8 requests at once | 87 tokens/s (mlx-lm) | 271 tokens/s | 3.1x |
| 8 agents asking at once: time to the first word | 42 s (Ollama) | 1.6 s | 26x sooner |
| One agent rewriting a 100-line file | 43 tokens/s (mlx-lm) | 128 tokens/s | 3.0x |
| An agent's first request after a restart | 16 s (mlx-lm) | 1.7 s | 9.6x sooner |
| A follow-up question about a 48,000-token document | out of memory (mlx-lm) | 0.6 s | |
| 16 coding agents calling tools (Qwen3.5-2B, from the paper) | 75 s (mlx-lm) | 29 s | 2.5x |
| 40 programming problems from LiveCodeBench, 8 at a time (from the paper) | 20 min (mlx-lm) | 8 min | 2.5x |

<div dir="ltr">
<img alt="Qwen3.5-4B through the server, 1 to 16 requests at once. M2 Pro · 16 GB: mlx-lm [51, 79, 88, 75, 124], Locals [65, 116, 150, 244, 310]. M2 Max · 96 GB: mlx-lm [91, 121, 142, 152, 267], Locals [104, 177, 244, 403, 530]. M5 Pro · 24 GB: mlx-lm [86, 151, 232, 258, 333], Locals [90, 161, 253, 382, 508]." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/macs.svg#gh-light-mode-only" width="100%">
<img alt="Qwen3.5-4B through the server, 1 to 16 requests at once. M2 Pro · 16 GB: mlx-lm [51, 79, 88, 75, 124], Locals [65, 116, 150, 244, 310]. M2 Max · 96 GB: mlx-lm [91, 121, 142, 152, 267], Locals [104, 177, 244, 403, 530]. M5 Pro · 24 GB: mlx-lm [86, 151, 232, 258, 333], Locals [90, 161, 253, 382, 508]." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/macs-dark.svg#gh-dark-mode-only" width="100%">
</div>

تحقق النماذج الأكبر حجماً مكاسب أعلى على معالج M2 Max: تبلغ 3.0x لنموذج Qwen3.5-9B و 3.1x لنموذج 27B مع معالجة ثمانية طلبات في آنٍ واحد. وعلى معالج M5 Pro تكون المكاسب أقل، نظراً لأن mlx-lm أسرع بكثير هناك: حيث تمتلك MLX نويات مخصصة للمسرعات العصبية في M5، وقد تم قياس هذه الأرقام قبل أن يمتلك Locals نوياته الخاصة لها. أما رقاقات M1 و M3 و M4 فلم يتم قياسها بعد: نرحب بنتائجها عبر [قسم المشكلات (Issues)](https://github.com/Generated-Computers/Locals/issues)، مرفقة بناتج الأمرين `locals bench` و `locals tune --report`.

<a id="benchmarks"></a>
## الاختبارات القياسية

كافة الجداول وتفاصيل قياس كل منها متاحة في صفحة
[الاختبارات القياسية (Benchmarks)](https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md). وما لم يذكر الرسم خلاف ذلك، فقد أُجريت القياسات على جهاز MacBook Pro بمعالج M2 Pro (معالج رسوميات بـ 19 نواة) وذاكرة 16 جيجابايت، باستخدام نموذج Qwen3.5-4B (`mlx-community/Qwen3.5-4B-MLX-4bit`)، وإصدار MLX 0.32 و mlx-lm 0.32.

<div dir="ltr">
<img alt="M2 Pro · 16 GB: Qwen3.5-2B 3.1x (188 to 576 tokens/s), MiniCPM5-2B 3.0x (160 to 478 tokens/s), LFM2.5-2.6B 2.8x (144 to 404 tokens/s), Qwen3.5-4B 3.1x (87 to 271 tokens/s), Gemma 4 E4B 2.8x (80 to 224 tokens/s). M2 Max · 96 GB: Qwen3.5-4B 2.66x (152 to 403 tokens/s), Qwen3.5-9B 3.03x (72 to 220 tokens/s), Qwen3.8-27B 3.13x (25 to 78 tokens/s). M5 Pro · 24 GB: Qwen3.5-4B 1.48x (258 to 382 tokens/s), Qwen3.5-9B 1.58x (145 to 229 tokens/s)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/models.svg#gh-light-mode-only" width="100%">
<img alt="M2 Pro · 16 GB: Qwen3.5-2B 3.1x (188 to 576 tokens/s), MiniCPM5-2B 3.0x (160 to 478 tokens/s), LFM2.5-2.6B 2.8x (144 to 404 tokens/s), Qwen3.5-4B 3.1x (87 to 271 tokens/s), Gemma 4 E4B 2.8x (80 to 224 tokens/s). M2 Max · 96 GB: Qwen3.5-4B 2.66x (152 to 403 tokens/s), Qwen3.5-9B 3.03x (72 to 220 tokens/s), Qwen3.8-27B 3.13x (25 to 78 tokens/s). M5 Pro · 24 GB: Qwen3.5-4B 1.48x (258 to 382 tokens/s), Qwen3.5-9B 1.58x (145 to 229 tokens/s)." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/models-dark.svg#gh-dark-mode-only" width="100%">
</div>

تحقق النماذج المُكمّمة إلى 4 بت بالطريقة المعتادة (في مجموعات من 64) تسريعاً يتراوح بين 2.3x إلى 3.3x مع معالجة 8 طلبات في آنٍ واحد؛ بينما تعمل التنسيقات الأخرى بنفس سرعة mlx-lm ([انظر أدناه](#where-locals-is-not-faster)).

<a id="against-other-servers"></a>
### مقارنة بالخوادم الأخرى

<div dir="ltr">
<img alt="Qwen3.5-4B, 1,000-token prompts, 8 requests at once on an M2 Pro. Time to the first token: Locals 1.6 s, llama.cpp 2.1 s, Magnitude 16.1 s, Ollama 41.8 s. Tokens per second: Locals 110, llama.cpp 71, Magnitude 42, Ollama 34." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/servers.svg#gh-light-mode-only" width="100%">
<img alt="Qwen3.5-4B, 1,000-token prompts, 8 requests at once on an M2 Pro. Time to the first token: Locals 1.6 s, llama.cpp 2.1 s, Magnitude 16.1 s, Ollama 41.8 s. Tokens per second: Locals 110, llama.cpp 71, Magnitude 42, Ollama 34." src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/charts/servers-dark.svg#gh-dark-mode-only" width="100%">
</div>

مقارنة بين Ollama 0.40.5 و Magnitude 0.2.9 و llama.cpp `b11724`، طلباً تلو الآخر على معالج الرسوميات، كلٌّ ببنائه الخاص بنسخة 4 بت من النموذج، ومقاسة باستخدام عميل InferenceX التابع لـ SemiAnalysis (مع درجة حرارة 0 وردود ذات طول طبيعي). تتوفر جداول مطالبات 1,000 و 8,000 رمز ونموذج Gemma 4 في صفحة [الاختبارات القياسية (Benchmarks)](https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md#3-against-other-servers).

<a id="long-prompts"></a>
### المطالبات الطويلة

**دون توقف مؤقت أثناء قراءة المطالبات الطويلة.** عندما يرسل وكيل مطالبة بحجم 8,000 رمز بينما يتلقى آخرون إجاباتهم، يقرأها Locals في نفس دورات إنتاج الرموز التالية لهم، مما يحافظ على استمرار تدفق ردودهم (streaming): تقلصت أطول فجوة في الردود من 12 ثانية إلى 3 ثوانٍ فقط، وبنفس معدل المعالجة الإجمالي، مقارنة بقراءة المطالبة منفردة. والطلب بحجم 3,000 رمز الذي يصل أثناء معالجة طلب بحجم 30,000 رمز يحصل على رمزه الأول بعد 7 ثوانٍ بدلاً من 41 ثانية.

<a id="where-locals-is-not-faster"></a>
### الحالات التي لا يكون فيها Locals أسرع

- **طلب فردي بمفرده، دون وجود ما يمكن إعادة استخدامه:** السرعة متقاربة تقريباً مع mlx-lm (64 مقابل 61 رمزاً في الثانية على M2 Pro). ويكون أسرع منفرداً عندما يكرر الرد أجزاء من المطالبة (مثل إعادة كتابة الملفات) أو عند استخدام التخزين المؤقت للمطالبة (prompt caching).
- **المهام المقيدة بالمطالبة (Prompt-bound work):** قراءة المطالبة تكون مقيدة بالقدرة الحسابية لمعالج الرسوميات. مع مطالبات بحجم 8,000 رمز و 8 طلبات متزامنة، يكتب الخادم 57 رمزاً في الثانية؛ حيث يُستهلك معظم الوقت في عملية القراءة.
- **التنسيقات التي تقع خارج النويات المضبوطة:** التكميم التآلفي 4 بت في مجموعات من 64 مغطى بالكامل؛ أما تنسيق mxfp4 (نموذج خليط الخبراء Mellum) فيعمل بنفس سرعة mlx-lm.
- **المسرعات العصبية في معالجات M5:** نويات MLX الخاصة تستخدمها بالفعل، لذا فإن مكاسب السرعة هناك تبلغ 1.5x وليس 3x (قبل نواة `nax` الخاصة بـ Locals، والتي لا تزال تجريبية).

<a id="works-with-your-agents"></a>
## متوافق مع وكلائك

<div dir="ltr">
<p align="center">
<img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/hermes.png" width="12.4%" alt="Hermes Agent" title="Hermes Agent"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/kilo.png" width="12.4%" alt="Kilo Code" title="Kilo Code"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/opencode.png" width="12.4%" alt="OpenCode" title="OpenCode"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/openhands.png" width="12.4%" alt="OpenHands" title="OpenHands"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/cline.png" width="12.4%" alt="Cline" title="Cline"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/aider.png" width="12.4%" alt="Aider" title="Aider"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/roo.png" width="12.4%" alt="Roo Code" title="Roo Code"><img src="https://raw.githubusercontent.com/Generated-Computers/Locals/main/docs/assets/agents/openclaw.png" width="12.4%" alt="OpenClaw" title="OpenClaw">
</p>
</div>

يعمل Locals مع وكلاء ومساعدي البرمجة الذين يستخدمهم المطورون بالفعل، على نموذج يعمل مباشرة على جهاز Mac الخاص بك. تم اختبار جميع هذه الأدوات باستثناء Roo Code بشكل شامل ومتكامل (end-to-end) مع نموذج Qwen3.5-4B على جهاز Mac بذاكرة 16 جيجابايت (كلاً من Cline و Kilo Code عبر إصدارات سطر الأوامر الخاصة بهما)، على مهام برمجية حقيقية مثل كتابة برنامج وتشغيله وتقديم تقرير بما قام بطباعته. أمر واحد يكفي لتشغيل Locals وتوصيل الأداة به:

| الأداة | أمر التشغيل |
|---|---|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | `locals launch hermes` |
| [Kilo Code](https://github.com/Kilo-Org/kilocode) | `locals launch kilo` |
| [OpenCode](https://github.com/anomalyco/opencode) | `locals code ~/my-project` |
| [OpenHands](https://github.com/OpenHands/OpenHands) | `locals launch openhands` |
| [Cline](https://github.com/cline/cline) | `locals launch cline` |
| [Aider](https://github.com/Aider-AI/aider) | `locals launch aider` |
| [Roo Code](https://github.com/RooCodeInc/Roo-Code) | في VS Code، أدخل الإعدادات من `locals launch editor` ([الكيفية](#use-it-in-apps-and-editors)) |
| [OpenClaw](https://github.com/openclaw/openclaw) | `locals launch openclaw` |
| [Claude Code](https://github.com/anthropics/claude-code) | `locals launch claude` (من خلال واجهة Anthropic Messages API التي يدعمها Locals أيضاً) |

يعرض الأمر `locals launch list` كافة الأدوات التي يمكنه فتحها. ويواصل Locals العمل في الخلفية بعد إغلاق الأداة: يُظهر الأمر `locals status` ما يقوم به، ويوقفه الأمر `locals stop`.

<sub>جميع الشعارات ملك لأصحابها ومشاريعها وتُعرض فقط للدلالة على توافق Locals مع هذه الأدوات.</sub>

<a id="getting-started"></a>
## البدء والاستخدام

تحتاج إلى جهاز Mac بمعالج Apple silicon (M1 أو أحدث) ونظام macOS 14 أو أحدث. يمنحك Mac بذاكرة 16 جيجابايت أفضل النتائج؛ ومع 8 جيجابايت، استخدم أحد النماذج الأصغر.

افتح تطبيق الطرفية (Terminal) وشغّل:

</div>

```bash
curl -LsSf https://raw.githubusercontent.com/Generated-Computers/Locals/main/install.sh | sh
```

<div dir="rtl" lang="ar">

ثم افتح نافذة جديدة في الطرفية (يقوم برنامج التثبيت بإضافة Locals إلى مسار PATH لصدفتك، والذي لا تقرأه سوى النوافذ الجديدة) وابدأ تشغيل مساعد البرمجة في أحد مشاريعك:

</div>

```bash
locals code ~/my-project
```

<div dir="rtl" lang="ar">

بعد التشغيل الأول الذي يقوم بتنزيل النموذج، يبدأ Locals بالعمل خلال ثوانٍ معدودة.

- **OpenCode** يعتمد Locals كنموذجه الافتراضي من تلك اللحظة فصاعداً، لذا فإن تشغيل `opencode` العادي سيستخدمه أيضاً؛ ويعيد الأمر `locals disconnect opencode` الإعدادات الأصلية الخاصة بـ OpenCode.
- **للتحديث:** شغّل أمر التثبيت مرة أخرى، أو نفّذ `uv tool upgrade locals`.
- **للإزالة:** شغّل `locals stop`، ثم `locals disconnect opencode`، وأخيراً `uv tool uninstall locals`.
- **مع مدير الحزم uv:** يمكنك التثبيت من PyPI عبر `uv tool install --python 3.12 locals`.

<a id="coming-from-ollama-lm-studio-mlx-lm-or-llamacpp"></a>
### الانتقال من Ollama أو LM Studio أو mlx-lm أو llama.cpp

يثبت Locals بجانبها ولا يغير أي شيء من إعداداتها: فله منفذه الخاص (8321) ومجلداته المستقلة. الذاكرة فقط هي المشتركة، لذا يُنصح بإنهاء التطبيق الآخر أثناء تشغيل Locals، إذ نادراً ما تتسع الذاكرة لنموذجين في وقت واحد. ويتذكر Locals النموذج والمنفذ اللذين بدأت بهما للأوامر اللاحقة.

**Ollama.** أغلق Ollama، ثم دع Locals يجيب عبر منفذ Ollama. التطبيقات المضبوطة لـ Ollama (مثل Open WebUI و Raycast و Continue) وأمر `ollama` نفسه تستمر في العمل دون تغيير، أياً كان اسم النموذج المطلوب:

</div>

```bash
osascript -e 'quit app "Ollama"'      # or: brew services stop ollama
locals launch --model qwen3.5:4b ollama
```

<div dir="rtl" lang="ar">

تعمل أسماء نماذج Ollama في Locals، وبالنسبة لأي نموذج لا يعرفه Locals، فإنه يقترح نسخة MLX المقابلة له على Hugging Face لاستخدامها بدلاً منه. تنزيلات Ollama تكون بصيغتها الخاصة (GGUF لمعظم النماذج)، لذا يُنزّل النموذج مرة ثانية بصيغة MLX. للعودة، شغّل `locals stop` ثم شغّل Ollama.

**LM Studio.** تعمل نماذج MLX الخاصة بـ LM Studio في Locals في مكانها دون الحاجة لتنزيل ثانٍ. أوقف خادم LM Studio وشغّل Locals على منفذه، لكي تستمر التطبيقات التي تستخدمه في العمل:

</div>

```bash
locals start --model ~/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit --port 1234
```

<div dir="rtl" lang="ar">

**mlx-lm.** نفس النماذج ونفس مجلد التنزيل، لذا لا يُنزّل أي شيء مرة أخرى. يحل `locals serve` محل `mlx_lm.server`، مع نفس واجهة برمجة التطبيقات (API):

</div>

```bash
locals serve --model mlx-community/Qwen3.5-4B-MLX-4bit --port 8080
```

<div dir="rtl" lang="ar">

**llama.cpp.** يقوم Locals بتشغيل نماذج MLX وليس ملفات GGUF؛ وإذا أعطيته ملفاً، فإنه يحدد نسخة MLX المقابلة من نفس النموذج. شغّله على منفذ `llama-server` للاحتفاظ بعملاء الاتصال لديك:

</div>

```bash
locals start --model qwen --port 8080
```

<div dir="rtl" lang="ar">

**أي خادم آخر يدعم واجهة برمجة تطبيقات OpenAI** (مثل vLLM أو LocalAI): وجّه عميلك إلى `http://127.0.0.1:8321/v1`، أو ابدأ Locals على المنفذ الذي يستخدمه العميل بالفعل عبر `--port`.

<a id="examples"></a>
## أمثلة

**العمل على مشروع مع مساعد برمجي.** يفتح هذا الأمر OpenCode في مشروعك، حيث يمكنك طلب التغييرات بكلمات بسيطة ومشاهدته وهو يقرأ ويعدل ويختبر الكود البرمجي:

</div>

```bash
locals code ~/my-project
```

<div dir="rtl" lang="ar">

**إسناد مهمة واحدة والحصول على النتيجة.** يقوم المساعد بالمهمة من البداية إلى النهاية دون فتح نافذة، ثم يطبع ما أنجزه:

</div>

```bash
locals code ~/my-project --run "add a --verbose flag to cli.py and run the tests"
```

<div dir="rtl" lang="ar">

**دع فريقاً من الوكلاء ينجز مهمة أكبر.** يعمل عدة وكلاء على أجزاء مختلفة من المهمة في الوقت نفسه، كلٌّ في نسخته المستقلة من مشروعك، واختباراتك البرمجية هي التي تقرر متى تكتمل المهمة. لا يتم تعديل مشروعك إلا عند اجتياز جميع الاختبارات:

</div>

```bash
locals run "write wordcount.py with a function count_words(text) and tests in test_wordcount.py" \
    --dir ~/my-project --check "python3 -m pytest -q"
```

<div dir="rtl" lang="ar">

**اطرح سؤالاً حول الكود البرمجي.** يقرأ الوكيل المشروع ويطبع الإجابة:

</div>

```bash
locals run "Which function parses dates, and which format does it expect?" --dir ~/my-project
```

<div dir="rtl" lang="ar">

**المحادثة مع النموذج** في الطرفية، كبرنامج محادثة شخصي وخاص:

</div>

```bash
locals chat
```

<div dir="rtl" lang="ar">

**تحويل النصوص إلى بيانات مهيكلة.** حدد الحقول التي تريدها في مخطط JSON Schema، على سبيل المثال في ملف يسمى `person.json`:

</div>

```json
{"type": "object",
 "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string"},
                "status": {"enum": ["pending", "approved", "rejected"]}, "tags": {"type": "array", "items": {"type": "string"}}},
 "required": ["name", "age", "email", "status", "tags"]}
```

<div dir="rtl" lang="ar">

عندئذٍ تلتزم الإجابة بالمخطط دائماً، حقلاً بحقل:

</div>

```bash
locals generate --prompt "Ana Silva, 29, ana@example.com, application approved, tags: vip, early" \
    --schema person.json
```

```json
{"name":"Ana Silva","age":29,"email":"ana@example.com","status":"approved","tags":["vip","early"]}
```

<div dir="rtl" lang="ar">

<a id="use-it-in-apps-and-editors"></a>
## استخدامه في التطبيقات والمحررات

ابدأ تشغيل Locals مرة واحدة بالنموذج الذي تريده؛ وسيواصل العمل في الخلفية:

</div>

```bash
locals start --model qwen
```

<div dir="rtl" lang="ar">

**تطبيق OpenCode.** شغّل الأمر `locals setup --model qwen` مرة واحدة، ثم اختر Locals من قائمة نماذج OpenCode. يعمل هذا في تطبيق سطح المكتب، وإضافة المحرر، وإصدار الطرفية على حد سواء.

**إضافات Kilo Code و Cline و Roo Code في VS Code**، و **Continue و Zed و JetBrains.** اختر المزود المسمى "OpenAI Compatible" وأدخل القيم التالية. شغّل الأمر `locals launch editor` لطباعة القيم الخاصة بالنموذج قيد التشغيل حالياً.

| الإعداد | القيمة المدخلة |
|---|---|
| Base URL | `http://127.0.0.1:8321/v1` |
| API key | أي نص، مثلاً `local` |
| Model | اسم النموذج من الجدول أدناه، مثلاً `locals/qwen3.5-4b` |
| Context window | الرقم الذي يطبعه الأمر `locals launch editor`، إذا طلبت الإضافة ذلك |

**التطبيقات المتوافقة مع Ollama**، مثل Open WebUI أو Raycast، لا تحتاج إلى أي إعدادات إضافية؛ فقط شغّل: `locals launch ollama`.

<a id="models"></a>
## النماذج

يختار Locals تلقائياً نموذجاً يلائم سعة ذاكرة جهاز Mac الخاص بك، لذا لا يحتاج أغلب المستخدمين إلى تغييره. لاستخدام نموذج آخر، ابدأ Locals باسمه، مثلاً `locals start --model lfm`. يشغّل Locals نموذجاً واحداً في كل مرة. العمودان الأخيران يوضحان السرعات بعدد الرموز في الثانية (tps) مع معالجة 8 طلبات في آنٍ واحد، على جهاز M2 Pro بذاكرة 16 جيجابايت المذكور أعلاه.

| النموذج | بدء التشغيل عبر | الاسم في محررك | الحجم | mlx-lm | Locals |
|---|---|---|---|---|---|
| Qwen3.5-4B (الافتراضي) | `--model qwen` | `locals/qwen3.5-4b` | 2.8 GB | 87 tps | 271 tps |
| LFM2.5-2.6B | `--model lfm` | `locals/lfm2.5-2.6b` | 1.5 GB | 144 tps | 404 tps |
| MiniCPM5-2B | `--model minicpm` | `locals/minicpm5-2b` | 1.3 GB | 160 tps | 478 tps |
| Gemma 4 E4B | `--model gemma` | `locals/gemma-4-e4b-it` | 4.8 GB | 80 tps | 224 tps |

على أجهزة Mac التي تمتلك أقل من 12 جيجابايت من الذاكرة، يكون النموذج الافتراضي هو Qwen3.5-2B بدلاً من ذلك (`--model qwen3.5-2b`؛ و `locals/qwen3.5-2b` في محررك).

يحتوي كل نموذج أيضاً على وضع التفكير (thinking mode)، وفيه يقوم بالاستدلال المنطقي والتفكير قبل الإجابة: أضف `-thinking` إلى الاسم، مثلاً `locals/lfm2.5-2.6b-thinking`. تُعد النماذج الأصغر خيارات ممتازة لأجهزة Mac المزودة بذاكرة 8 جيجابايت (8 GB)، كما تعمل أيضاً النماذج الأخرى المصممة لمعالجات Apple silicon على [Hugging Face](https://huggingface.co/mlx-community).

يمكن أن تكون المحادثات طويلة جداً: حتى 178,000 رمز تقريباً مع النموذج الافتراضي على Mac بذاكرة 16 جيجابايت، أي ما يعادل بضع مئات من صفحات النصوص. على جهاز M2 Pro، تستغرق قراءة مستند بحجم 170,000 رمز حوالي 12 دقيقة في المرة الأولى، وتبدأ أسئلة المتابعة حوله بعد ثانيتين فقط. تم ضبط OpenCode لـ 32,000 رمز، وكل من OpenClaw و Hermes Agent و Kilo Code لـ 64,000 رمز، مما يحافظ على سرعة كل خطوة من خطواتها.

<a id="using-it-from-your-own-code"></a>
## استخدامه من خلال برمجياتك الخاصة

يفهم Locals نفس الطلبات التي تستقبلها خدمات OpenAI و Ollama، لذا فإن البرامج المكتوبة لها تعمل مع Locals دون أي تعديل. ابدأ تشغيل Locals عبر `locals start`، ثم وجّه برنامجك إلى `http://127.0.0.1:8321/v1`:

</div>

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8321/v1", api_key="local")
reply = client.chat.completions.create(
    model="locals",  # the model the server runs
    messages=[{"role": "user", "content": "Write a haiku about the sea."}],
)
print(reply.choices[0].message.content)
```

<div dir="rtl" lang="ar">

أرسل العديد من الطلبات في وقت واحد، سواء عبر خيوط المعالجة المتعددة (threads) أو باستخدام عميل غير متزامن (async client)، وسيقوم Locals بالإجابة عليها مجتمعة؛ وهنا تتجلى أقصى سرعاته. كما يعمل التدفق (streaming)، والردود بصيغة JSON محددة، واستدعاء الدوال (function calling)، ومكتبة Ollama، وأمر `curl` البسيط بنفس الكفاءة؛ وتوضح [الأمثلة](https://github.com/Generated-Computers/Locals/blob/main/examples/README.md) كلاً منها بالتفصيل.

<a id="privacy-and-safety"></a>
## الخصوصية والأمان

كل شيء يعمل محلياً على جهاز Mac الخاص بك. لا يقبل Locals سوى الطلبات الصادرة من التطبيقات الموجودة على حاسوبك فقط، وبالتالي لا يمكن لأي موقع ويب تزوره استخدامه. وعندما يعمل فريق من الوكلاء على مهمة (`locals run`، `locals solve`)، فإن الأوامر التي ينفذونها يمكنها تعديل الملفات فقط في نسختهم المعزولة من مشروعك، وفي مجلدات الملفات المؤقتة والذاكرة المؤقتة التي تكتب إليها الأدوات. أما `locals code` فيشغل OpenCode في مشروعك مباشرة، بالطريقة الاعتيادية التي يعمل بها OpenCode دائماً.

<a id="if-something-does-not-work"></a>
## إذا واجهتك مشكلة

شغّل الأمر `locals status`، أو افتح الرابط http://127.0.0.1:8321 في متصفحك، لترى ما إذا كان Locals قيد التشغيل وما الذي يفعله حالياً. على المعالجات الأخرى غير M2 Pro و M2 Max (المضمن ضبطهما مسبقاً مع Locals)، يقوم البرنامج بضبط نفسه تلقائياً لمعالجك في المرة الأولى، في الخلفية أثناء فترات خمول الجهاز: ما بين 10 إلى 20 دقيقة من وقت الخمول إجمالاً. يمكنك استخدامه خلال ذلك الوقت، وتزداد سرعته فور اكتمال الضبط.

نرحب بالأسئلة والملاحظات والمشكلات عبر [قسم المشكلات على GitHub (Issues)](https://github.com/Generated-Computers/Locals/issues).

<a id="documentation"></a>
## التوثيق

| الموضوع | الصفحة |
|---|---|
| آلية عمل Locals: الخادم، ومحرك الدفعات، والذاكرة المؤقتة للمطالبات، وفريق الوكلاء خلف `locals run`، وكافة الإعدادات | [كيف يعمل Locals](https://github.com/Generated-Computers/Locals/blob/main/docs/README.md) |
| كافة الاختبارات القياسية، وكيفية قياسها | [الاختبارات القياسية (Benchmarks)](https://github.com/Generated-Computers/Locals/blob/main/docs/benchmarks/README.md) |
| كافة الجداول مع تواريخها، وإصدارات التثبيت (Commits)، والأوامر؛ ومعالج M2 Max | [القياسات (Measurements)](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/README.md)، [M2 Max](https://github.com/Generated-Computers/Locals/blob/main/docs/measurements/m2-max.md) |
| أسباب تفوق سرعة Locals على mlx-lm بالتفصيل | [لماذا يعد Locals أسرع](https://github.com/Generated-Computers/Locals/blob/main/docs/why-faster/README.md) |
| التدفق (Streaming)، وصيغ JSON، واستدعاء الدوال، ومكتبة Ollama، وأمر `curl` | [أمثلة (Examples)](https://github.com/Generated-Computers/Locals/blob/main/examples/README.md) |
| الاختبارات، وحزمة الاختبارات الشاملة (E2E) مع أدوات الوكلاء الحقيقية | [الاختبارات (Tests)](https://github.com/Generated-Computers/Locals/blob/main/tests/README.md)، [الاختبارات الشاملة (End to end)](https://github.com/Generated-Computers/Locals/blob/main/tests/e2e/README.md) |
| المساهمة في تطوير Locals نفسه | [دليل المساهمة (CONTRIBUTING.md)](https://github.com/Generated-Computers/Locals/blob/main/CONTRIBUTING.md) |

للعمل على تطوير Locals:

</div>

```bash
git clone https://github.com/Generated-Computers/Locals
cd Locals
uv sync --extra dev
uv run pytest tests -q -m "not model"
```

<div dir="rtl" lang="ar">

<a id="citation"></a>
## الاستشهاد الأكاديمي

إذا استخدمت Locals في بحثك العلمي، يرجى الاستشهاد بالورقة البحثية:

</div>

```bibtex
@article{asadulaev2026breaking,
  title   = {Breaking the Space Barrier and its Application to Language Model Inference},
  author  = {Asadulaev, Arip},
  journal = {arXiv preprint arXiv:2610.09139},
  year    = {2026}
}
```

<div dir="rtl" lang="ar">

<a id="license"></a>
## الترخيص

مشروع Locals مرخص بموجب رخصة MIT. طُوّر بواسطة
[Generated Computers](https://github.com/Generated-Computers).

</div>
