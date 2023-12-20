package com.capstone.fitfriendapp.data.response

import com.google.gson.annotations.SerializedName
data class RegisterResponse(
    @field:SerializedName("error")
    val error: String,

    @field:SerializedName("message")
    val message: String,
)