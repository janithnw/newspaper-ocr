# Analysis of Jíbaro and Campesino Mentions in Spanish Newspapers

This repository contains Jupyter notebooks and Python utilities for analyzing historical Spanish-language newspapers from Puerto Rico, using OCR text from the Library of Congress's Chronicling America collection. The analysis focuses on mentions of *jíbaro*, *gíbaro*, and *campesino*, including spelling and grammatical variants.

The numbered notebooks in the repository root guide the workflow from downloading and extracting text to correcting OCR errors and filtering irrelevant matches. They produce timelines of mentions by year and newspaper, and word clouds of the words and phrases surrounding those mentions.

# Approach

1. **Collect and extract the text.** Download existing newspaper OCR data from Chronicling America and extract text blocks from ALTO XML files. Order lines within each block and join words split by end-of-line hyphenation, retaining publication dates, newspaper identifiers, and links to the original pages.

2. **Clean OCR errors.** Use GPT to produce conservative corrections for a sample of pages, preserving the original Spanish and avoiding speculative changes. Compare these examples with the raw text to identify recurring word and phrase corrections. Combine these patterns with Spanish vocabulary resources, corpus word frequencies, and spelling-correction heuristics to clean the full collection.

3. **Find candidate mentions.** Search the cleaned text for spelling and grammatical variants of *jíbaro*, *gíbaro*, and *campesino*, including accented and unaccented forms. Retain matching text blocks and extract windows of surrounding words to help distinguish the different uses of each term.

4. **Filter irrelevant matches.** Manually label examples and train separate classifiers for *jíbaro* and *gíbaro*, using TF-IDF features from both the full text block and the surrounding words. Examples labeled as irrelevant include shipping notices and Cuban news referring to Gibara as a place name, literary titles or publication listings such as *La maldición de un jíbaro*, commercial advertisements, and miscellaneous notices. Calibrated linear support vector machines extend the manual judgments to unlabeled blocks, while manual labels take precedence where available. Mentions of *campesino* are retained based on keyword matching without a separate relevance classifier.

5. **Generate timelines.** Count mentions in the retained text blocks, aggregate them by year and newspaper, and plot the resulting timelines. Compare *jíbaro* and *gíbaro* mentions, individually or combined, with *campesino* mentions.

6. **Generate word clouds.** Gather sentences or short passages surrounding the target terms in the retained texts. Use Spanish language processing and dictionary filtering to identify nouns, adjectives, and recurring two- and three-word phrases. Remove stop words, the target terms themselves, and uninformative terms (such as reptitions, plurals, and persistent OCR errors), then size the remaining words and phrases by frequency.
