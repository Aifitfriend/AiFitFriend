package com.capstone.fitfriendapp.view.home

import android.content.Context
import android.content.Intent
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat.startActivity
import androidx.recyclerview.widget.RecyclerView
import com.capstone.fitfriendapp.R

class ListWorkOutAdapter(private val listWorkOut: ArrayList<WorkOut>) :
    RecyclerView.Adapter<ListWorkOutAdapter.ListViewHolder>() {
    private var context: Context? = null

    private var onItemClickCallback: OnItemClickCallback? = null

    fun setOnItemClickCallback(onItemClickCallback: OnItemClickCallback) {
        this.onItemClickCallback = onItemClickCallback
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ListViewHolder {
        val view: View = LayoutInflater.from(parent.context).inflate(R.layout.item, parent, false)
        return ListViewHolder(view)
    }

    override fun onAttachedToRecyclerView(recyclerView: RecyclerView) {
        super.onAttachedToRecyclerView(recyclerView)
        context = recyclerView.context
    }

    override fun onBindViewHolder(holder: ListViewHolder, position: Int) {
        val (name, photo) = listWorkOut[position]
        holder.imgPhoto.setImageResource(photo)
        holder.tvName.text = name

        holder.itemView.setOnClickListener {
            onItemClickCallback?.onItemClicked(listWorkOut[holder.adapterPosition])
            context?.let {ctx ->
                val intent: Intent? = when (name) {
                    "Upper Body" -> Intent(context, UpperBodyActivity::class.java)
                    "Arm Exercise" -> null
                    "Leg Exercise" -> null
                    else -> null
                }
                intent?.let {intent ->
                    startActivity(ctx, intent, null) }
            }
        }
    }

    override fun getItemCount() = listWorkOut.size


    class ListViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imgPhoto: ImageView = itemView.findViewById(R.id.iv_item)
        val tvName: TextView = itemView.findViewById(R.id.tv_item)

    }


    interface OnItemClickCallback {
        fun onItemClicked(data: WorkOut)
    }


}