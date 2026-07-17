# Case Study: Preference Alignment Example (LaMP-7)

**Task:** LaMP-7 tweet-style paraphrase (user-personalized)
**Model:** LLaMA3-8b-Instruct
**Sample ID:** 560

## User Profile (excerpt)
- I am officially bored and there is nothin' to watch 
- @LydiaBarling did u not get my txt i sent it at midnight 
- @ericjstar i didnt even sleep last night lol didnt even take a nap  and the storm hit lol

## Input
Today, I am finding it difficult to concentrate on my assessment test as I am distracted by the desire to get a new tattoo and piercing. I am amused by the situation and attribute it to my attention deficit disorder.

## Reference (gold)
I want a new tattoo...and piercing!  I need to keep focused on this assesment test LMFAO!!!!!! I have A-D-D today 

## Outputs
**DPS:** "I'm having a hard time focusing on my test cuz I wanna get a new tattoo and piercing so bad, lol"
**Baseline:** "I'm so bored, there's nothing to watch"
**CAD:** "I'm so bored, there's nothing to watch"
**DeCoReVanilla:** "I'm so bored there's nothing to watch"
**DoLa:** "I'm so bored, nothing to watch"

## Why this demonstrates preference alignment
- The gold response uses casual tweet style with contractions and slang ("I want", "LMFAO", "A-D-D").
- DPS preserves the meaning (distracted by tattoo/piercing during a test) and matches the informal style ("cuz", "wanna", "lol").
- Baselines ignore the input and copy an unrelated profile snippet ("bored, nothing to watch"), showing weaker alignment to the current user intent.

# Case Study: Preference Alignment Example (LaMP-7)

**Task:** LaMP-7 tweet-style paraphrase (user-personalized)
**Model:** LLaMA3-8b-Instruct
**Sample ID:** 189

## User Profile (excerpt)
- @YoungQ Morning Rob!!  Blessings and have a great day too!! 
- @CrunchyK Pics please!!  I love the set-up pics 
- @YoungQ Have a good night and sleep well!! Thanks so much for all you do for us and God Bless! HUGS 

## Input
@YoungQ sends their love and wishes for a great show. They apologize for preferring Vanilla Stoli iced coffees tonight.

## Reference (gold)
@YoungQ Much love!!! Have a great show!! And sorry but I'm into Vanilla Stoli iced coffees tonight - so yummy! 

## Outputs
**DPS:** "@YoungQ Sending love and good vibes for a great show tonight! Sorry, but I'm sticking with Vanilla Stoli iced coffees"
**Baseline:** "@YoungQ Morning Rob!!  Blessings and have a great day too!! "
**CAD:** "@YoungQ Morning Rob!!  Blessings and have a great day too!! "
**DeCoReVanilla:** "@YoungQ Morning Rob!!  Blessings and have a great day too!! "
**DoLa:** "@YoungQ Morning Rob!!  Blessings and have a great day too!! "

## Why this demonstrates preference alignment
- DPS preserves the addressee (@YoungQ) and the key preference content (Vanilla Stoli iced coffees), matching the gold and user style (enthusiastic tone).
- Baselines copy an unrelated profile line (addressed to “Rob”), which conflicts with the current input and shows weak alignment.
