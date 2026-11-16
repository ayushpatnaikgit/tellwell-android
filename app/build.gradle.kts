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
        versionCode = 3
        versionName = "0.3.0"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_static") }
        }
    }

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
}
