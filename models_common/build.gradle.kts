plugins { id("com.android.asset-pack") }

assetPack {
    packName.set("models_common")
    dynamicDelivery { deliveryType.set("fast-follow") }
}
