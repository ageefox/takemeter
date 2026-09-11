# Dataset notes

`takemeter_dataset.csv` contains 212 short excerpts collected from publicly accessible discussion threads on WetCanvas and KnittingHelp. The dataset was assembled for a small, non-commercial NLP study of discourse in art and craft communities.

Each row records:

- a project-specific numeric ID
- the excerpt text
- a manually assigned discourse label
- the source site, thread URL, and thread title
- a short collection note

Usernames and account identifiers are not included. The source URLs are retained so examples can be traced back to their original context.

The repository's MIT license applies to the code, not to the forum excerpts. Rights in the source text remain with the original authors and platforms. Anyone reusing the dataset should review the applicable source terms and determine whether their use is appropriate.

The labels reflect one annotator's judgment. They have not been independently adjudicated, and some boundaries—especially `technical_help` versus `critique_feedback`—depend on conversational context that is absent from the excerpt.
