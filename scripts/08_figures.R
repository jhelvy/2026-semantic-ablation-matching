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
ablation_levels <- c('Scrambled', 'Ciphered Words', 'Ciphered Letters')

# Method lineup (PLAN.md, "Method lineup"): semantic methods are the object of
# study and go in the main figures; TF-IDF is the lexical control, drawn there
# as a reference. All lexical methods go in the appendix (*_lexical.png).
# The v1 LLM runs (results/v1_llm/) are superseded and labeled "(v1)".
semantic_methods <- c('llm', 'llm_haiku', 'openai_embeddings', 'sentence_transformer')
lexical_methods <- c('tfidf', 'soft_tfidf', 'monge_elkan', 'levenshtein', 'jaro_winkler')

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
        method == 'llm_sonnet55' ~ 'Claude Sonnet 5.5',
        method == 'llm_gpt4omini' ~ 'GPT-4o-mini',
        method == 'llm_kimik3' ~ 'Kimi K3',
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
    facet_grid(method ~ dataset) +
    scale_color_manual(values = plotColors) +
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
    mutate(transformation = factor(transformation, levels = names(plotColors)))

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
    scale_color_manual(values = plotColors) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
    labs(x = 'Recall', y = 'Precision', color = NULL) +
    theme_minimal_grid(font_family = font, font_size = 16)
  p <- facet_theme(p) + theme(legend.position = 'bottom') + panel_border()
  p
}

pr_plot(cv_v1 |> filter(method_raw %in% semantic_methods), tfidf_points = TRUE)
save_fig('precision_recall_scatter.png', width = 8, height = 10)

pr_plot(cv_v1 |> filter(method_raw %in% lexical_methods))
save_fig('precision_recall_scatter_lexical.png', width = 8, height = 12)

# Plot 3: Robustness check — GPT-4o vs GPT-4o-mini (Amazon-Google only) ---------
# v1 results. Shows all transformations as points + error bars by model tier.

robustness_lookup <- tribble(
  ~method     , ~tier              ,
  'llm'       , 'GPT-4o-mini (v1)' ,
  'llm_gpt4o' , 'GPT-4o (v1)'      ,
)

cv_robustness <- cv_raw |>
  inner_join(robustness_lookup, by = 'method') |>
  filter(dataset == 'amazon-google', transformation %in% v1_conditions) |>
  mutate(
    dataset = str_replace_all(dataset, "-", " ") |> str_to_title(),
    transformation = str_replace_all(transformation, "_", " ") |>
      str_to_title(),
    tier = factor(tier, levels = robustness_lookup$tier)
  )

robustness_baseline <- cv_robustness |>
  filter(transformation == 'Original') |>
  select(tier, dataset, f1_baseline = f1_mean, sd_baseline = f1_std)

robustness_ablations <- cv_robustness |>
  filter(transformation != 'Original') |>
  mutate(transformation = factor(transformation, levels = ablation_levels)) |>
  left_join(robustness_baseline, by = c('tier', 'dataset')) |>
  mutate(
    pct_label = if_else(
      is.na(f1_mean),
      NA_character_,
      sprintf('%+.0f%%', (f1_mean - f1_baseline) / f1_baseline * 100)
    )
  )

robustness_ablations |>
  ggplot(aes(x = f1_mean, y = transformation)) +
  geom_rect(
    data = robustness_baseline,
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
  geom_vline(data = robustness_baseline, aes(xintercept = f1_baseline)) +
  geom_vline(
    data = tfidf_ref |> filter(dataset == 'Amazon Google'),
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
    linewidth = 0.5
  ) +
  geom_point(size = 2.5) +
  geom_text(
    aes(label = pct_label, x = f1_mean - f1_std),
    hjust = 1.1,
    vjust = 0.5,
    size = 3,
    color = 'grey20',
    family = font,
    show.legend = FALSE
  ) +
  facet_grid(tier ~ dataset) +
  scale_x_continuous(
    limits = c(0, 1),
    expand = expansion(add = c(0.12, 0.02))
  ) +
  labs(
    x = 'F1 Score',
    y = 'Transformation',
    caption = 'Dashed red line: TF-IDF (lexical control) F1 on Original'
  ) +
  coord_cartesian(clip = 'off') +
  theme_minimal_vgrid(font_family = font, font_size = 14) +
  theme(
    strip.background = element_rect(fill = 'grey80', color = NA),
    panel.grid.minor = element_blank(),
    plot.title.position = 'plot',
    plot.caption.position = 'plot',
    legend.position = 'none',
    panel.background = element_rect(fill = 'white', color = NA),
    plot.background = element_rect(fill = 'white', color = NA)
  ) +
  panel_border()

save_fig('llm_robustness.png', width = 8, height = 6)

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
