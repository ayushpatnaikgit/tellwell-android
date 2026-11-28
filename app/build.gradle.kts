import java.util.Properties

plugins {
    id("com.android.application")
}

// Release signing comes from outside the repository: set tellwell.keystore and tellwell.keystorePasswordFile
// in local.properties (never committed). Without them, only debug builds can be made.
val local = Properties().apply {
    rootProject.file("local.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) }
}
val keystore = local.getProperty("tellwell.keystore")?.let { file(it) }
val keystorePassword = local.getProperty("tellwell.keystorePasswordFile")?.let { file(it).readText().trim() }

android {
    namespace = "io.github.ayushpatnaikgit.tellwell"
    compileSdk = 36
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "io.github.ayushpatnaikgit.tellwell"
        minSdk = 29
        targetSdk = 36
        versionCode = 11
        versionName = "0.3.8"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_static") }
        }
    }

    // Two builds of the same app. github: downloads its models itself (internet permission, used once). play: Google
    // Play delivers the models as asset packs and the app has no internet permission.
    flavorDimensions += listOf("store", "edition")
    productFlavors {
        create("github") { dimension = "store" }
        create("play") { dimension = "store" }
        // Two apps from the same code. english: TellWell (English only). multilingual: TellWell Multilingual, a
        // separate app where the user adds languages (Hindi for now); each language downloads its own models.
        create("english") {
            dimension = "edition"
            buildConfigField("boolean", "MULTILINGUAL", "false")
        }
        create("multilingual") {
            dimension = "edition"
            applicationIdSuffix = ".multilingual"
            buildConfigField("boolean", "MULTILINGUAL", "true")
        }
    }
    // Only Play app bundles use these; the GitHub APK doesn't include them.
    assetPacks += listOf(":models_common", ":models_cpu", ":models_npu")

    signingConfigs {
        if (keystore != null && keystorePassword != null) {
            create("release") {
                storeFile = keystore
                storePassword = keystorePassword
                keyAlias = "tellwell"
                keyPassword = keystorePassword
            }
        }
    }

    buildTypes {
        debug {
            // Test builds signed with the release key install over the released app without wiping its models.
            signingConfigs.findByName("release")?.let { signingConfig = it }
        }
        release {
            // No code shrinking: the native libraries call into these classes by name.
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release")
        }
    }

    buildFeatures { buildConfig = true }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    // Native libraries are extracted to disk: the NPU loads its code (libggml-htp-*.so) from a real directory.
    packaging {
        jniLibs {
            useLegacyPackaging = true
            // Qualcomm's runtime: keep the NPU (HTP) libraries only, not the older DSP or the GPU backends.
            excludes += listOf("**/libQnnDsp*.so", "**/libQnnGpu.so")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // Qualcomm AI Engine Direct runtime (QNN) for speech recognition on the NPU. Its licence allows distribution only
    // inside an app, so it comes from Maven at build time and is never committed (see THIRD_PARTY_qnn-runtime_NOTICE.txt).
    implementation("com.qualcomm.qti:qnn-runtime:2.42.0")
    // Play Asset Delivery (Google Play build only): asks the Play Store for the model packs.
    "playImplementation"("com.google.android.play:asset-delivery:2.3.0")
}
