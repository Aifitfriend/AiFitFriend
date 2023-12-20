package com.capstone.fitfriendapp.data.pref
data class UserModel(
    val email: String,
    val token: String,
    val isLogin: Boolean = false
)