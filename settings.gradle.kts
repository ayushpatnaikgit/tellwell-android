pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "TellWell"
include(":app")
// Google Play build only: the models, delivered by Play as asset packs (files fetched by scripts/fetch-play-models.sh).
include(":models_common", ":models_cpu", ":models_npu")
