library(tidyverse)
library(cowplot)
library(ggh4x)

options(dplyr.width = Inf)

# Paths are anchored to the repo root (this script lives in scripts/), so it
# runs from any working directory. Reads results/cv_results.csv, writes figs/.
script_arg <- grep('^--file=', commandArgs(trailingOnly = FALSE), value = TRUE)
repo_root <- if (length(script_arg) == 1) {
  normalizePath(file.path(dirname(sub('^--file=', '', script_arg)), '..'))
} else {
  getwd()
}
results_dir <- file.path(repo_root, 'results')
figs_dir <- file.path(repo_root, 'figs')
dir.create(figs_dir, showWarnings = FALSE)

# Under Rscript the default device is pdf(), which rejects the Roboto Condensed
# font when cowplot draws grobs ("invalid font type"). Draw on an off-screen
# ragg device instead; ggsave() below also uses ragg.
ragg::agg_png(tempfile(fileext = '.png'))

# Plot settings ---------------------------------------------------------------

font <- 'Roboto Condensed'

# Transformation colors, plus the TF-IDF reference drawn in every main figure.
plotColors <- c(
  'Original' = 'grey20',
  'Ciphered Letters' = '#80C5DCFF',
  'Ciphered Words' = '#E37D39FF',
  'Scrambled' = '#8FC977FF'
)
tfidf_color <- '#980000FF'

# The four v1 conditions shown in the overview figures. The factorial cells
# (desc_ciphered, ids_ciphered, all_ciphered) get their own figure later.
v1_conditions <- c('original', 'ciphered_letters', 'ciphered_words', 'scrambled')
# Two groups of ablations (decided 2026-11-10): Ciphered Letters and Scrambled can
# be read by a capable model (decode test, 09_decode_test.py), so they are
# "meaning readable"; Ciphered Words removes meaning. Levels run bottom to top,
# so the meaning-removed condition sits at the bottom, below a dotted divider.
ablation_levels <- c('Ciphered Words', 'Scrambled', 'Ciphered Letters')
ablation_labels <- c(
  'Ciphered Letters' = 'Ciphered Letters\n(readable)',
  'Scrambled' = 'Scrambled\n(readable)',
  'Ciphered Words' = 'Ciphered Words\n(meaning removed)'
)
condition_legend <- c(
  'Original' = 'Original',
  'Ciphered Letters' = 'Ciphered Letters (readable)',
  'Ciphered Words' = 'Ciphered Words (meaning removed)',
  'Scrambled' = 'Scrambled (readable)'
)

# Method lineup (PLAN.md, "Method lineup"): semantic methods are the object of
# study and go in the main figures; TF-IDF is the lexical control, drawn there
# as a reference. All lexical methods go in the appendix (*_lexical.png).
# The main figures show the small LLMs (Haiku 5.5, GPT-6 Luna). Sonnet 5.5 is
# a scale check and appears only in the appendix figure llm_scale_check.png.
# Until the v2 small models have run, the superseded v1 LLM runs
# (results/v1_llm/) stand in, labeled "(v1)".
v2_small_llms <- c('llm_haiku55', 'llm_gpt6luna')
v1_main_llms <- c('llm', 'llm_haiku')
embedding_methods <- c('openai_embeddings', 'sentence_transformer')
lexical_methods <- c('tfidf', 'soft_tfidf', 'monge_elkan', 'levenshtein', 'jaro_winkler')

# Scale check: every model scored on Sonnet's fixed 30% of queries
# (05_cv_threshold.py writes the small models and TF-IDF as "*__sonnet30").
scale_methods <- c(
  'llm_haiku55__sonnet30' = 'Claude Haiku 5.5',
  'llm_gpt6luna__sonnet30' = 'GPT-6 Luna',
  'llm_sonnet55' = 'Claude Sonnet 5.5'
)
scale_colors <- c(
  'Claude Haiku 5.5' = 'grey65',
  'GPT-6 Luna' = 'grey30',
  'Claude Sonnet 5.5' = '#2166ac'
)
all_conditions <- c(
  original = 'Original',
  ciphered_letters = 'Ciphered Letters',
  ciphered_words = 'Ciphered Words',
  scrambled = 'Scrambled',
  desc_ciphered = 'Descriptions ciphered',
  ids_ciphered = 'Identifiers ciphered',
  all_ciphered = 'Both ciphered'
)

# Shared label cleaning --------------------------------------------------------

clean_labels <- function(d) {
  d |>
    mutate(
      method = case_when(
        method == 'llm' ~ 'GPT-4o-mini\n(v1)',
        method == 'llm_gpt4o_mini_token_fallback' ~ 'GPT-4o-mini\nToken Fallback (v1)',
        method == 'llm_haiku' ~ 'Claude Haiku\n(v1)',
        method == 'llm_haiku_token_fallback' ~ 'Claude Haiku\nToken Fallback (v1)',
        method == 'llm_claude' ~ 'Claude Sonnet\n(v1)',
        method == 'llm_gpt4o' ~ 'GPT-4o\n(v1)',
        method == 'llm_haiku55' ~ 'Claude Haiku 5.5',
        method == 'llm_gpt6luna' ~ 'GPT-6 Luna',
        method == 'llm_sonnet55' ~ 'Claude Sonnet 5.5',
        method == 'tfidf' ~ 'TF-IDF',
        method == 'soft_tfidf' ~ 'Soft TF-IDF',
        method == 'openai_embeddings' ~ 'OpenAI Emb.',
        method == 'sentence_transformer' ~ 'Transformer',
        .default = str_replace_all(method, "_", " ") |> str_to_title()
      ),
      dataset = str_replace_all(dataset, "-", " ") |> str_to_title(),
      transformation = str_replace_all(transformation, "_", " ") |>
        str_to_title()
    )
}

# Order methods by their best F1 across conditions and datasets.
order_methods <- function(d) {
  method_order <- d |>
    group_by(method) |>
    summarise(top_f1 = max(f1_mean), .groups = 'drop') |>
    arrange(desc(top_f1)) |>
    pull(method)
  d |> mutate(method = factor(method, levels = method_order))
}

facet_theme <- function(base) {
  base +
    theme(
      strip.background = element_rect(fill = 'grey80', color = NA),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = 'white', color = NA),
      plot.background = element_rect(fill = 'white', color = NA)
    )
}

save_fig <- function(name, width, height) {
  ggsave(
    device = ragg::agg_png, # base png() device fails on the Roboto Condensed font
    filename = file.path(figs_dir, name),
    width = width,
    height = height,
    dpi = 300
  )
}

# Data -------------------------------------------------------------------------

cv_raw <- read_csv(file.path(results_dir, 'cv_results.csv'), show_col_types = FALSE)

main_llms <- if (any(cv_raw$method %in% v2_small_llms)) v2_small_llms else v1_main_llms
semantic_methods <- c(main_llms, embedding_methods)

cv_v1 <- cv_raw |>
  filter(transformation %in% v1_conditions) |>
  mutate(method_raw = method) |>
  clean_labels()

# TF-IDF is invariant to the transforms (within about 0.01 F1), so one
# reference per dataset at its Original F1 is enough in the F1 figures.
tfidf_ref <- cv_v1 |>
  filter(method_raw == 'tfidf', transformation == 'Original') |>
  select(dataset, tfidf_f1 = f1_mean)

# Plot 1: F1 by transformation, with 5-fold CV error bars (mean +/- 1 SD) -------
# Each method's own Original F1 is the solid line (grey band = +/- 1 SD);
# points are the three ablations, labeled with the % change from Original.

f1_plot <- function(d, tfidf_line = FALSE) {
  d <- order_methods(d)
  baseline <- d |>
    filter(transformation == 'Original') |>
    select(method, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)
  ablations <- d |>
    filter(transformation != 'Original') |>
    mutate(transformation = factor(transformation, levels = ablation_levels)) |>
    left_join(baseline, by = c('method', 'dataset')) |>
    mutate(
      pct_label = sprintf('%+.0f%%', (f1_mean - f1_baseline) / f1_baseline * 100)
    )

  p <- ablations |>
    ggplot(aes(x = f1_mean, y = transformation, color = transformation)) +
    geom_rect(
      data = baseline,
      inherit.aes = FALSE,
      aes(
        xmin = f1_baseline - sd_baseline,
        xmax = f1_baseline + sd_baseline,
        ymin = -Inf,
        ymax = Inf
      ),
      fill = 'grey70',
      alpha = 0.35
    ) +
    geom_vline(data = baseline, aes(xintercept = f1_baseline))

  if (tfidf_line) {
    # Label the reference once, in the top row of panels.
    tfidf_label <- tfidf_ref |>
      mutate(method = factor(levels(d$method)[1], levels = levels(d$method)))
    p <- p +
      geom_vline(
        data = tfidf_ref,
        aes(xintercept = tfidf_f1),
        color = tfidf_color,
        linetype = 'dashed',
        linewidth = 0.6
      ) +
      geom_text(
        data = tfidf_label,
        inherit.aes = FALSE,
        aes(x = tfidf_f1, y = Inf, label = 'TF-IDF'),
        color = tfidf_color,
        family = font,
        size = 3.5,
        hjust = 1.1,
        vjust = 1.3
      )
  }

  p <- p +
    geom_errorbar(
      orientation = 'y',
      aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
      width = 0.3,
      alpha = 0.9,
      linewidth = 0.5
    ) +
    geom_point(size = 1.8) +
    geom_text(
      aes(label = pct_label, x = f1_mean - f1_std),
      hjust = 1.1,
      vjust = 0.5,
      size = 2.8,
      color = 'grey20',
      family = font,
      show.legend = FALSE
    ) +
    geom_hline(yintercept = 1.5, color = 'grey50', linetype = 'dotted') +
    facet_grid(method ~ dataset) +
    scale_color_manual(values = plotColors) +
    scale_y_discrete(labels = ablation_labels) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      limits = c(0, 1),
      expand = expansion(add = c(0.12, 0.02))
    ) +
    coord_cartesian(clip = 'off') +
    labs(x = 'F1 Score', y = 'Transformation') +
    theme_minimal_vgrid(font_family = font, font_size = 16)
  p <- facet_theme(p) + theme(legend.position = 'none') + panel_border()
  p
}

# Main text: semantic methods, TF-IDF as a dashed reference line.
f1_plot(cv_v1 |> filter(method_raw %in% semantic_methods), tfidf_line = TRUE)
save_fig('f1_comparison.png', width = 8, height = 8)

# Appendix: lexical methods (TF-IDF is an ordinary row here).
f1_plot(cv_v1 |> filter(method_raw %in% lexical_methods))
save_fig('f1_comparison_lexical.png', width = 8, height = 9.5)

# Plot 2: Precision-recall scatter ----------------------------------------------
# Cross-validated operating point from cv_results.csv, consistent with the F1
# figure.

pr_plot <- function(d, tfidf_points = FALSE) {
  d <- d |>
    order_methods() |>
    mutate(transformation = factor(
      transformation,
      levels = c('Original', 'Ciphered Letters', 'Scrambled', 'Ciphered Words')
    ))

  p <- d |>
    ggplot() +
    geom_point(
      aes(x = recall, y = precision, color = transformation),
      size = 2
    )

  if (tfidf_points) {
    # TF-IDF in every panel, one point per condition (they nearly coincide,
    # because TF-IDF is invariant). Labeled once, in the top row.
    tfidf_pr <- cv_v1 |>
      filter(method_raw == 'tfidf') |>
      select(dataset, transformation, precision, recall)
    tfidf_label <- tfidf_pr |>
      filter(transformation == 'Original') |>
      mutate(method = factor(levels(d$method)[1], levels = levels(d$method)))
    p <- p +
      geom_point(
        data = tfidf_pr,
        aes(x = recall, y = precision),
        color = tfidf_color,
        shape = 18,
        size = 3
      ) +
      geom_text(
        data = tfidf_label,
        aes(x = recall, y = precision, label = 'TF-IDF'),
        color = tfidf_color,
        family = font,
        size = 3.5,
        hjust = 1.25,
        vjust = 0.5
      )
  }

  p <- p +
    facet_grid(method ~ dataset) +
    scale_color_manual(values = plotColors, labels = condition_legend) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(x = 'Recall', y = 'Precision', color = NULL) +
    theme_minimal_grid(font_family = font, font_size = 16)
  p <- facet_theme(p) +
    theme(legend.position = 'bottom') +
    guides(color = guide_legend(nrow = 2)) +
    panel_border()
  p
}

pr_plot(cv_v1 |> filter(method_raw %in% semantic_methods), tfidf_points = TRUE)
save_fig('precision_recall_scatter.png', width = 8, height = 10)

pr_plot(cv_v1 |> filter(method_raw %in% lexical_methods))
save_fig('precision_recall_scatter_lexical.png', width = 8, height = 12)

# Plot 3 (the v1 GPT-4o vs GPT-4o-mini robustness figure, llm_robustness.png)
# was removed 2026-11-10: candidate_list_robustness.png (Plot 7) replaces it.

# Plot 4: Per-model LLM prompt comparison (standard vs token-fallback) ----------
# v1 results. One row per model family, faceted by dataset. Ablations on the
# y-axis, dodged points + error bars colored by prompt variant. Original
# (standard prompt) is the solid reference line; TF-IDF is dashed.

llm_lookup <- tribble(
  ~method                         , ~model              , ~prompt          ,
  'llm'                           , 'GPT-4o-mini (v1)'  , 'Standard'       ,
  'llm_gpt4o_mini_token_fallback' , 'GPT-4o-mini (v1)'  , 'Token-Fallback' ,
  'llm_haiku'                     , 'Claude Haiku (v1)' , 'Standard'       ,
  'llm_haiku_token_fallback'      , 'Claude Haiku (v1)' , 'Token-Fallback' ,
)

cv_llm <- cv_raw |>
  inner_join(llm_lookup, by = 'method') |>
  filter(transformation %in% v1_conditions) |>
  mutate(
    dataset = str_replace_all(dataset, "-", " ") |> str_to_title(),
    transformation = str_replace_all(transformation, "_", " ") |>
      str_to_title(),
    prompt = factor(prompt, levels = c('Standard', 'Token-Fallback'))
  )

llm_baseline <- cv_llm |>
  filter(transformation == 'Original', prompt == 'Standard') |>
  select(model, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)

llm_ablations <- cv_llm |>
  filter(transformation != 'Original') |>
  mutate(transformation = factor(transformation, levels = ablation_levels)) |>
  left_join(llm_baseline, by = c('model', 'dataset')) |>
  mutate(
    pct_label = sprintf('%+.0f%%', (f1_mean - f1_baseline) / f1_baseline * 100)
  )

llm_ablations |>
  ggplot(aes(x = f1_mean, y = transformation, color = prompt)) +
  geom_rect(
    data = llm_baseline,
    inherit.aes = FALSE,
    aes(
      xmin = f1_baseline - sd_baseline,
      xmax = f1_baseline + sd_baseline,
      ymin = -Inf,
      ymax = Inf
    ),
    fill = 'grey70',
    alpha = 0.35
  ) +
  geom_vline(data = llm_baseline, aes(xintercept = f1_baseline)) +
  geom_vline(
    data = tfidf_ref,
    aes(xintercept = tfidf_f1),
    color = tfidf_color,
    linetype = 'dashed',
    linewidth = 0.6
  ) +
  geom_errorbar(
      orientation = 'y',
    aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
    width = 0.3,
    alpha = 0.9,
    linewidth = 0.5,
    position = position_dodge(width = 0.5)
  ) +
  geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
  geom_text(
    aes(label = pct_label, x = f1_mean - f1_std, group = prompt),
    hjust = 1.1,
    vjust = 0.5,
    size = 2.8,
    color = 'grey20',
    family = font,
    show.legend = FALSE,
    position = position_dodge(width = 0.5)
  ) +
  facet_grid(model ~ dataset) +
  scale_x_continuous(
    limits = c(0, 1),
    expand = expansion(add = c(0.14, 0.02))
  ) +
  scale_color_manual(
    values = c('Standard' = '#2166ac', 'Token-Fallback' = '#E37D39FF')
  ) +
  labs(
    x = 'F1 Score',
    y = 'Transformation',
    color = 'Prompt',
    caption = 'Dashed red line: TF-IDF (lexical control) F1 on Original'
  ) +
  coord_cartesian(clip = 'off') +
  theme_minimal_vgrid(font_family = font, font_size = 14) +
  theme(
    strip.background = element_rect(fill = 'grey80', color = NA),
    panel.grid.minor = element_blank(),
    plot.title.position = 'plot',
    plot.caption.position = 'plot',
    legend.position = 'bottom',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  panel_border()

save_fig('llm_prompt_comparison.png', width = 8, height = 6)

# Plot 5 (appendix): scale check — Sonnet 5.5 vs the small models ---------------
# All three models on the same queries: Sonnet's fixed 30% subsample. Every
# condition, v1 transforms and 2x2 factorial cells. TF-IDF (dashed) is scored on
# the same queries. Drawn only once Sonnet results exist.

cv_scale <- cv_raw |>
  filter(method %in% names(scale_methods), transformation %in% names(all_conditions))

if (!any(cv_scale$method == 'llm_sonnet55')) {
  message('llm_scale_check.png skipped: no llm_sonnet55 results yet')
} else {
  cv_scale <- cv_scale |>
    mutate(
      model = factor(scale_methods[method], levels = scale_methods),
      dataset = str_replace_all(dataset, "-", " ") |> str_to_title(),
      group = case_when(
        transformation == 'original' ~ 'Baseline',
        transformation %in% v1_conditions ~ 'v1 transforms',
        .default = '2x2 factorial'
      ),
      group = factor(group, levels = c('Baseline', 'v1 transforms', '2x2 factorial')),
      # Reversed, so Original sits at the top of each panel.
      transformation = factor(all_conditions[transformation], levels = rev(all_conditions))
    )

  tfidf_scale_ref <- cv_raw |>
    filter(method == 'tfidf__sonnet30', transformation == 'original') |>
    mutate(dataset = str_replace_all(dataset, "-", " ") |> str_to_title()) |>
    select(dataset, tfidf_f1 = f1_mean)
  tfidf_scale_label <- tfidf_scale_ref |>
    mutate(group = factor('Baseline', levels = levels(cv_scale$group)))

  dodge <- position_dodge(width = 0.7, orientation = 'y', reverse = TRUE)

  cv_scale |>
    ggplot(aes(x = f1_mean, y = transformation, color = model)) +
    geom_vline(
      data = tfidf_scale_ref,
      aes(xintercept = tfidf_f1),
      color = tfidf_color,
      linetype = 'dashed',
      linewidth = 0.6
    ) +
    geom_text(
      data = tfidf_scale_label,
      inherit.aes = FALSE,
      aes(x = tfidf_f1, y = Inf, label = 'TF-IDF'),
      color = tfidf_color,
      family = font,
      size = 3.5,
      hjust = 1.1,
      vjust = 1.3
    ) +
    geom_errorbar(
      orientation = 'y',
      aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
      width = 0.4,
      linewidth = 0.5,
      position = dodge
    ) +
    geom_point(size = 2.2, position = dodge) +
    facet_grid(group ~ dataset, scales = 'free_y', space = 'free_y') +
    scale_color_manual(values = scale_colors) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    labs(
      x = 'F1 Score',
      y = NULL,
      color = NULL,
      caption = paste(
        "All models scored on the same queries: Sonnet's fixed 30% subsample.",
        'Error bars: +/- 1 SD across 5 CV folds.\nDashed red line: TF-IDF (lexical control) F1 on Original, same queries.'
      )
    ) +
    theme_minimal_vgrid(font_family = font, font_size = 14) +
    theme(
      strip.background = element_rect(fill = 'grey80', color = NA),
      panel.grid.minor = element_blank(),
      plot.caption.position = 'plot',
      plot.caption = element_text(hjust = 0),
      legend.position = 'bottom',
      panel.background = element_rect(fill = 'white', color = NA),
      plot.background = element_rect(fill = 'white', color = NA)
    ) +
    panel_border()

  save_fig('llm_scale_check.png', width = 8, height = 7)
}

# Plot 6: 2x2 factorial — descriptive words x identifiers ----------------------
# Interaction plot: x = identifiers intact or ciphered, one line per description
# state. The key contrast is "Descriptions ciphered, identifiers intact": if a
# model used the preserved identifiers, that point would stay near Original.
# Main lineup (small LLMs + embeddings); TF-IDF (invariant across cells) is the
# dashed reference.

factorial_cells <- tribble(
  ~transformation , ~descriptions , ~identifiers ,
  'original'      , 'Intact'      , 'Intact'     ,
  'desc_ciphered' , 'Ciphered'    , 'Intact'     ,
  'ids_ciphered'  , 'Intact'      , 'Ciphered'   ,
  'all_ciphered'  , 'Ciphered'    , 'Ciphered'   ,
)
desc_colors <- c('Intact' = 'grey20', 'Ciphered' = '#E37D39FF')

cv_factorial <- cv_raw |>
  filter(method %in% semantic_methods) |>
  inner_join(factorial_cells, by = 'transformation') |>
  mutate(method_raw = method) |>
  clean_labels() |>
  order_methods() |>
  mutate(
    identifiers = factor(identifiers, levels = c('Intact', 'Ciphered')),
    descriptions = factor(descriptions, levels = c('Intact', 'Ciphered'))
  )

if (n_distinct(cv_factorial$transformation) < 4) {
  message('factorial_2x2.png skipped: factorial cells missing from cv_results.csv')
} else {
  # Direct labels for the two description lines, top row only.
  factorial_label <- cv_factorial |>
    filter(method == levels(method)[1], dataset == 'Abt Buy', identifiers == 'Ciphered') |>
    mutate(label = paste('Descriptions', str_to_lower(descriptions)))
  factorial_dodge <- position_dodge(width = 0.15)

  cv_factorial |>
    ggplot(aes(x = identifiers, y = f1_mean, color = descriptions, group = descriptions)) +
    geom_hline(
      data = tfidf_ref,
      aes(yintercept = tfidf_f1),
      color = tfidf_color,
      linetype = 'dashed',
      linewidth = 0.6
    ) +
    geom_text(
      data = tfidf_ref |>
        mutate(method = factor(levels(cv_factorial$method)[1], levels = levels(cv_factorial$method))),
      inherit.aes = FALSE,
      aes(x = 0.5, y = tfidf_f1, label = 'TF-IDF'),
      color = tfidf_color,
      family = font,
      size = 3.5,
      hjust = 0,
      vjust = -0.5
    ) +
    geom_line(linewidth = 0.7, position = factorial_dodge) +
    geom_errorbar(
      aes(ymin = f1_mean - f1_std, ymax = f1_mean + f1_std),
      width = 0.1,
      linewidth = 0.5,
      position = factorial_dodge
    ) +
    geom_point(size = 2.2, position = factorial_dodge) +
    geom_text(
      data = factorial_label,
      aes(label = label),
      family = font,
      size = 3.3,
      hjust = 0,
      nudge_x = 0.12,
      show.legend = FALSE
    ) +
    facet_grid(method ~ dataset) +
    scale_color_manual(values = desc_colors) +
    scale_x_discrete(expand = expansion(add = c(0.5, 1.1))) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(
      x = 'Identifiers',
      y = 'F1 Score',
      caption = paste(
        'Error bars: +/- 1 SD across 5 CV folds.',
        'Dashed red line: TF-IDF (lexical control) F1, identical in every cell.'
      )
    ) +
    coord_cartesian(clip = 'off') +
    theme_minimal_hgrid(font_family = font, font_size = 14) +
    theme(
      strip.background = element_rect(fill = 'grey80', color = NA),
      panel.grid.minor = element_blank(),
      plot.caption.position = 'plot',
      plot.caption = element_text(hjust = 0),
      legend.position = 'none',
      panel.background = element_rect(fill = 'white', color = NA),
      plot.background = element_rect(fill = 'white', color = NA)
    ) +
    panel_border()

  save_fig('factorial_2x2.png', width = 8, height = 9)
}

# Plot 7: candidate-list robustness — ordered k=50 vs ordered k=10 -------------
# Haiku 5.5 and GPT-6 Luna, Original plus the three v1 ablations. Rank order is
# kept, so the runs differ only in list length (PLAN.md C.2 item 3). TF-IDF is
# the dashed reference.

k_lookup <- tribble(
  ~method              , ~model              , ~list      ,
  'llm_haiku55'        , 'Claude Haiku 5.5'  , 'k = 50'   ,
  'llm_haiku55__k10'   , 'Claude Haiku 5.5'  , 'k = 10'   ,
  'llm_gpt6luna'       , 'GPT-6 Luna'        , 'k = 50'   ,
  'llm_gpt6luna__k10'  , 'GPT-6 Luna'        , 'k = 10'   ,
)
k_colors <- c('k = 50' = '#2166ac', 'k = 10' = '#E37D39FF')

cv_k <- cv_raw |>
  inner_join(k_lookup, by = 'method') |>
  filter(transformation %in% v1_conditions) |>
  mutate(
    dataset = str_replace_all(dataset, "-", " ") |> str_to_title(),
    transformation = factor(
      all_conditions[transformation],
      levels = rev(c('Original', 'Ciphered Letters', 'Ciphered Words', 'Scrambled'))
    ),
    model = factor(model, levels = c('GPT-6 Luna', 'Claude Haiku 5.5')),
    list = factor(list, levels = names(k_colors))
  )

if (!any(str_detect(cv_k$method, '__k10'))) {
  message('candidate_list_robustness.png skipped: no __k10 results yet')
} else {
  k_dodge <- position_dodge(width = 0.5, orientation = 'y', reverse = TRUE)

  cv_k |>
    ggplot(aes(x = f1_mean, y = transformation, color = list)) +
    geom_vline(
      data = tfidf_ref,
      aes(xintercept = tfidf_f1),
      color = tfidf_color,
      linetype = 'dashed',
      linewidth = 0.6
    ) +
    geom_errorbar(
      orientation = 'y',
      aes(xmin = f1_mean - f1_std, xmax = f1_mean + f1_std),
      width = 0.3,
      linewidth = 0.5,
      position = k_dodge
    ) +
    geom_point(size = 2.2, position = k_dodge) +
    facet_grid(model ~ dataset) +
    scale_color_manual(values = k_colors) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    labs(
      x = 'F1 Score',
      y = NULL,
      color = 'Candidates shown (TF-IDF rank order)',
      caption = paste(
        'Error bars: +/- 1 SD across 5 CV folds.',
        'Dashed red line: TF-IDF (lexical control) F1 on Original.'
      )
    ) +
    theme_minimal_vgrid(font_family = font, font_size = 14) +
    theme(
      strip.background = element_rect(fill = 'grey80', color = NA),
      panel.grid.minor = element_blank(),
      plot.caption.position = 'plot',
      plot.caption = element_text(hjust = 0),
      legend.position = 'bottom',
      panel.background = element_rect(fill = 'white', color = NA),
      plot.background = element_rect(fill = 'white', color = NA)
    ) +
    panel_border()

  save_fig('candidate_list_robustness.png', width = 8, height = 6)
}
